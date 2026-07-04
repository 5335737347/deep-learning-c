#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"

/* ─── helpers ────────────────────────────────────────────────────────── */

static int64_t product(const int64_t *arr, int64_t n) {
    int64_t p = 1;
    for (int64_t i = 0; i < n; ++i) {
        if (p > 0 && arr[i] > INT64_MAX / p) return 0;
        p *= arr[i];
    }
    return p;
}

/* Increment a multi-dimensional index in row-major order (last dim fastest).
 * Returns false on overflow (all dimensions wrapped back to 0). */
static bool index_inc(int64_t *idx, const int64_t *shape, int64_t ndim) {
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (++idx[d] < shape[d]) return true;
        idx[d] = 0;
    }
    return false;
}

/* Allocate a combined shape+strides buffer (contiguous in memory). */
static int64_t *alloc_shape_strides(int64_t ndim, int64_t **strides_out) {
    int64_t *buf = (int64_t *)malloc((size_t)(2 * ndim) * sizeof(int64_t));
    if (buf != NULL && strides_out != NULL) *strides_out = buf + ndim;
    return buf;
}

/* Compute row-major strides into an already-allocated strides array. */
static void compute_strides(const int64_t *shape, int64_t *strides, int64_t ndim) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int64_t i = ndim - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
}

static void print_rec(const Tensor *t, int64_t dim, int64_t *idx) {
    if (dim == t->ndim) {
        float *val = tensor_at(t, idx);
        printf("%.6f", val ? *val : 0.0f);
        return;
    }
    printf("[");
    for (int64_t i = 0; i < t->shape[dim]; ++i) {
        if (i > 0) printf(" ");
        idx[dim] = i;
        print_rec(t, dim + 1, idx);
    }
    printf("]");
    if (dim == 0) printf("\n");
}

/* Copy elements in row-major order into dst_data (must be contiguous).
 * Uses carry-add index increment to avoid per-element division/modulo. */
static bool copy_flat(const Tensor *src, float *dst_data) {
    int64_t ndim = src->ndim;
    int64_t *idx = (int64_t *)calloc((size_t)ndim, sizeof(int64_t));
    if (idx == NULL) return false;

    for (int64_t i = 0; i < src->size; ++i) {
        float *s = tensor_at(src, idx);
        if (s == NULL) { free(idx); return false; }
        dst_data[i] = *s;
        index_inc(idx, src->shape, ndim);
    }
    free(idx);
    return true;
}

/* ─── storage ────────────────────────────────────────────────────────── */

Storage *storage_create(size_t n, bool zero_init, DeviceType device) {
    if (n == 0) {
        errno = EINVAL;
        return NULL;
    }
    if (device != DEVICE_CPU) {
        errno = ENOSYS;
        return NULL;
    }

    Storage *s = (Storage *)malloc(sizeof(Storage));
    if (s == NULL) return NULL;   /* errno set by malloc */

    s->data = zero_init ? (float *)calloc(n, sizeof(float))
                        : (float *)malloc(n * sizeof(float));
    if (s->data == NULL) {
        free(s);
        errno = ENOMEM;
        return NULL;
    }

    s->size      = n;
    s->ref_count = 1;
    s->device    = device;
    return s;
}

void storage_retain(Storage *s) {
    if (s == NULL) return;
    s->ref_count++;
}

void storage_release(Storage *s) {
    if (s == NULL) return;
    if (--s->ref_count == 0) {
        free(s->data);
        free(s);
    }
}

/* ─── tensor lifecycle ───────────────────────────────────────────────── */

Tensor *tensor_create(int64_t ndim, const int64_t *shape) {
    if (ndim <= 0 || shape == NULL) {
        errno = EINVAL;
        return NULL;
    }
    for (int64_t i = 0; i < ndim; i++) {
        if (shape[i] <= 0) { errno = EINVAL; return NULL; }
    }

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (t == NULL) return NULL;   /* errno set by malloc */

    t->ndim    = ndim;
    t->shape   = NULL;   /* combined shape+strides buffer, freed together */
    t->strides = NULL;
    t->storage = NULL;
    t->offset  = 0;
    t->size    = 0;

    t->shape = alloc_shape_strides(ndim, &t->strides);
    if (t->shape == NULL) { errno = ENOMEM; goto err_create; }

    memcpy(t->shape, shape, (size_t)ndim * sizeof(int64_t));
    compute_strides(shape, t->strides, ndim);

    t->size = product(shape, ndim);
    if (t->size == 0) { errno = ERANGE; goto err_shape; }

    t->storage = storage_create((size_t)t->size, true, DEVICE_CPU);
    if (t->storage == NULL) goto err_shape;   /* errno set by storage_create */

    return t;

err_shape:
    free(t->shape);   /* frees both shape and strides (single allocation) */
err_create:
    free(t);
    return NULL;
}

Tensor *tensor_create_view(Storage *storage, int64_t ndim,
                           const int64_t *shape, const int64_t *strides,
                           int64_t offset) {
    if (storage == NULL || ndim <= 0 || shape == NULL || strides == NULL) {
        errno = EINVAL;
        return NULL;
    }
    if (offset < 0 || (size_t)offset >= storage->size) {
        errno = EINVAL;
        return NULL;
    }
    for (int64_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) { errno = EINVAL; return NULL; }
    }

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (t == NULL) return NULL;   /* errno set by malloc */

    t->storage = storage;
    storage_retain(storage);
    t->ndim   = ndim;
    t->shape  = NULL;
    t->strides = NULL;
    t->offset = offset;
    t->size   = 0;

    t->shape = alloc_shape_strides(ndim, &t->strides);
    if (t->shape == NULL) { errno = ENOMEM; goto err_storage; }

    memcpy(t->shape,   shape,   (size_t)ndim * sizeof(int64_t));
    memcpy(t->strides, strides, (size_t)ndim * sizeof(int64_t));

    t->size = product(shape, ndim);
    if (t->size == 0) { errno = ERANGE; goto err_shape; }

    /* Verify the last logical element fits within the storage buffer.
     * Computes max flat index: offset + sum_{i} strides[i] * (shape[i] - 1).
     * For currently-all-positive strides this is the last element;
     * the check is safe as long as strides are non-negative. */
    {
        int64_t max_idx = t->offset;
        for (int64_t i = 0; i < t->ndim; ++i) {
            max_idx += t->strides[i] * (t->shape[i] - 1);
        }
        if (max_idx < 0 || (size_t)max_idx >= storage->size) {
            errno = EINVAL;
            goto err_shape;
        }
    }

    return t;

err_shape:
    free(t->shape);
err_storage:
    storage_release(storage);
    free(t);
    return NULL;
}

void tensor_free(Tensor *t) {
    if (t == NULL) return;
    storage_release(t->storage);
    free(t->shape);   /* shape and strides are a single allocation */
    free(t);
}

/* ─── element access ─────────────────────────────────────────────────── */

int64_t tensor_flat_index(const Tensor *t, const int64_t *indices) {
    if (t == NULL || indices == NULL) {
        errno = EINVAL;
        return 0;
    }
    int64_t idx = t->offset;
    for (int64_t i = 0; i < t->ndim; ++i) {
        idx += indices[i] * t->strides[i];
    }
    return idx;
}

float *tensor_at(const Tensor *t, const int64_t *indices) {
    if (t == NULL || t->storage == NULL || indices == NULL) {
        errno = EINVAL;
        return NULL;
    }
    int64_t idx = tensor_flat_index(t, indices);
    if (idx < 0 || (size_t)idx >= t->storage->size) {
        errno = EINVAL;
        return NULL;
    }
    return &t->storage->data[idx];
}

/* ─── queries ────────────────────────────────────────────────────────── */

bool tensor_is_contiguous(const Tensor *t) {
    if (t == NULL) return false;
    if (t->ndim == 0) return true;

    int64_t expected = 1;
    for (int64_t i = t->ndim; i > 0; --i) {
        if (t->strides[i - 1] != expected) return false;
        expected *= t->shape[i - 1];
    }
    return t->offset == 0;
}

/* ─── contiguous copy ────────────────────────────────────────────────── */

Tensor *tensor_contiguous(const Tensor *t) {
    if (t == NULL) { errno = EINVAL; return NULL; }

    if (tensor_is_contiguous(t)) {
        return tensor_create_view(t->storage, t->ndim,
                                  t->shape, t->strides, t->offset);
    }

    Tensor *result = tensor_create(t->ndim, t->shape);
    if (result == NULL) return NULL;   /* errno set by tensor_create */

    if (!copy_flat(t, result->storage->data)) {
        tensor_free(result);
        errno = ENOMEM;
        return NULL;
    }

    return result;
}

/* ─── clone ──────────────────────────────────────────────────────────── */

Tensor *tensor_clone(const Tensor *t) {
    if (t == NULL) { errno = EINVAL; return NULL; }

    Tensor *result = tensor_create(t->ndim, t->shape);
    if (result == NULL) return NULL;   /* errno set by tensor_create */

    if (!copy_flat(t, result->storage->data)) {
        tensor_free(result);
        errno = ENOMEM;
        return NULL;
    }

    return result;
}

/* ─── fill ───────────────────────────────────────────────────────────── */

void tensor_fill(Tensor *t, float value) {
    if (t == NULL) return;

    if (tensor_is_contiguous(t)) {
        for (int64_t i = 0; i < t->size; ++i)
            t->storage->data[t->offset + i] = value;
        return;
    }

    /* Non-contiguous: iterate via index carry-add */
    int64_t *idx = (int64_t *)calloc((size_t)t->ndim, sizeof(int64_t));
    if (idx == NULL) return;

    for (int64_t i = 0; i < t->size; ++i) {
        float *p = tensor_at(t, idx);
        if (p) *p = value;
        if (!index_inc(idx, t->shape, t->ndim) && i + 1 < t->size) break;
    }
    free(idx);
}

/* ─── transform ──────────────────────────────────────────────────────── */

Tensor *tensor_transpose(const Tensor *t, int64_t dim0, int64_t dim1) {
    if (t == NULL) { errno = EINVAL; return NULL; }
    if (dim0 < 0 || dim1 < 0 || dim0 >= t->ndim || dim1 >= t->ndim) {
        errno = EINVAL;
        return NULL;
    }

    /* No-op: transposing a dimension with itself */
    if (dim0 == dim1) {
        return tensor_create_view(t->storage, t->ndim,
                                  t->shape, t->strides, t->offset);
    }

    int64_t ndim = t->ndim;
    int64_t *buf = alloc_shape_strides(ndim, NULL);
    if (buf == NULL) return NULL;   /* errno set by malloc */
    int64_t *new_shape   = buf;
    int64_t *new_strides = buf + ndim;

    memcpy(new_shape,   t->shape,   (size_t)ndim * sizeof(int64_t));
    memcpy(new_strides, t->strides, (size_t)ndim * sizeof(int64_t));

    /* swap */
    int64_t tmp = new_shape[dim0];
    new_shape[dim0]   = new_shape[dim1];
    new_shape[dim1]   = tmp;
    tmp = new_strides[dim0];
    new_strides[dim0] = new_strides[dim1];
    new_strides[dim1] = tmp;

    Tensor *ret = tensor_create_view(t->storage, ndim,
                                     new_shape, new_strides, t->offset);
    free(buf);
    return ret;
}

Tensor *tensor_slice(const Tensor *t, int64_t dim, int64_t start, int64_t end) {
    if (t == NULL) { errno = EINVAL; return NULL; }
    if (dim < 0 || dim >= t->ndim || start < 0 || start >= end || end > t->shape[dim]) {
        errno = EINVAL;
        return NULL;
    }

    int64_t ndim = t->ndim;
    int64_t *new_shape = (int64_t *)malloc((size_t)ndim * sizeof(int64_t));
    if (new_shape == NULL) return NULL;   /* errno set by malloc */
    memcpy(new_shape, t->shape, (size_t)ndim * sizeof(int64_t));
    new_shape[dim] = end - start;
    int64_t new_offset = t->offset + start * t->strides[dim];

    Tensor *ret = tensor_create_view(t->storage, ndim,
                                     new_shape, t->strides, new_offset);
    free(new_shape);
    return ret;
}

Tensor *tensor_reshape(const Tensor *t, int64_t new_ndim,
                       const int64_t *new_shape) {
    if (t == NULL || new_shape == NULL) { errno = EINVAL; return NULL; }
    int64_t total = product(new_shape, new_ndim);
    if (total == 0)  { errno = ERANGE; return NULL; }
    if (total != t->size) { errno = EINVAL; return NULL; }

    if (tensor_is_contiguous(t)) {
        int64_t *new_strides = (int64_t *)malloc((size_t)new_ndim * sizeof(int64_t));
        if (new_strides == NULL) return NULL;   /* errno set by malloc */
        compute_strides(new_shape, new_strides, new_ndim);

        Tensor *ret = tensor_create_view(t->storage, new_ndim,
                                         new_shape, new_strides, t->offset);
        free(new_strides);
        return ret;
    } else {
        Tensor *cont = tensor_contiguous(t);
        if (cont == NULL) return NULL;   /* errno set by tensor_contiguous */
        Tensor *ret = tensor_reshape(cont, new_ndim, new_shape);
        tensor_free(cont);
        return ret;
    }
}

Tensor *tensor_broadcast_to(const Tensor *t, int64_t target_ndim,
                            const int64_t *target_shape) {
    if (t == NULL || target_shape == NULL) { errno = EINVAL; return NULL; }
    if (t->ndim > target_ndim) { errno = EINVAL; return NULL; }

    int64_t diff = target_ndim - t->ndim;
    int64_t *new_strides = (int64_t *)malloc((size_t)target_ndim * sizeof(int64_t));
    if (new_strides == NULL) return NULL;   /* errno set by malloc */

    for (int64_t i = 0; i < diff; ++i) new_strides[i] = 0;

    for (int64_t i = 0; i < t->ndim; ++i) {
        int64_t src_dim = t->shape[i];
        int64_t dst_dim = target_shape[i + diff];
        if (src_dim != 1 && src_dim != dst_dim) {
            free(new_strides);
            errno = EINVAL;
            return NULL;
        }
        new_strides[i + diff] = (src_dim == 1) ? 0 : t->strides[i];
    }

    Tensor *ret = tensor_create_view(t->storage, target_ndim,
                                     target_shape, new_strides, t->offset);
    free(new_strides);
    return ret;
}

Tensor *tensor_squeeze(const Tensor *t) {
    if (t == NULL) { errno = EINVAL; return NULL; }

    int64_t new_ndim = 0;
    for (int64_t i = 0; i < t->ndim; ++i) {
        if (t->shape[i] != 1) new_ndim++;
    }

    /* No size-1 dimensions: return a view of the original */
    if (new_ndim == t->ndim) {
        return tensor_create_view(t->storage, t->ndim,
                                  t->shape, t->strides, t->offset);
    }

    int64_t *buf = alloc_shape_strides(new_ndim, NULL);
    if (buf == NULL) return NULL;   /* errno set by malloc */
    int64_t *new_shape   = buf;
    int64_t *new_strides = buf + new_ndim;

    int64_t j = 0;
    for (int64_t i = 0; i < t->ndim; ++i) {
        if (t->shape[i] != 1) {
            new_shape[j]   = t->shape[i];
            new_strides[j] = t->strides[i];
            j++;
        }
    }

    Tensor *ret = tensor_create_view(t->storage, new_ndim,
                                     new_shape, new_strides, t->offset);
    free(buf);
    return ret;
}

Tensor *tensor_unsqueeze(const Tensor *t, int64_t dim) {
    if (t == NULL) { errno = EINVAL; return NULL; }
    if (dim < 0 || dim > t->ndim) { errno = EINVAL; return NULL; }

    int64_t new_ndim = t->ndim + 1;
    int64_t *buf = alloc_shape_strides(new_ndim, NULL);
    if (buf == NULL) return NULL;   /* errno set by malloc */
    int64_t *new_shape   = buf;
    int64_t *new_strides = buf + new_ndim;

    for (int64_t i = 0; i < dim; ++i) {
        new_shape[i]   = t->shape[i];
        new_strides[i] = t->strides[i];
    }
    new_shape[dim]   = 1;
    new_strides[dim] = 0;
    for (int64_t i = dim; i < t->ndim; ++i) {
        new_shape[i + 1]   = t->shape[i];
        new_strides[i + 1] = t->strides[i];
    }

    Tensor *ret = tensor_create_view(t->storage, new_ndim,
                                     new_shape, new_strides, t->offset);
    free(buf);
    return ret;
}

/* ─── debug ──────────────────────────────────────────────────────────── */

void tensor_print(const Tensor *t) {
    if (t == NULL) {
        printf("(null)\n");
        return;
    }
    if (t->ndim == 0) {
        if (t->storage && t->offset >= 0 &&
            (size_t)t->offset < t->storage->size) {
            printf("scalar: %.6f\n", t->storage->data[t->offset]);
        } else {
            printf("invalid scalar\n");
        }
        return;
    }
    int64_t *idx = (int64_t *)calloc((size_t)t->ndim, sizeof(int64_t));
    if (idx == NULL) {
        printf("(memory error)\n");
        return;
    }
    print_rec(t, 0, idx);
    free(idx);
}
