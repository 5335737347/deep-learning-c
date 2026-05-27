/**
 * @file tensor.c
 * @brief Implementation of the n-dimensional tensor data structure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "tensor.h"

Tensor *tensor_create(int ndim, const int *shape) {
    if (ndim <= 0 || shape == NULL) return NULL;

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (t == NULL) return NULL;

    t->data = NULL;
    t->shape = NULL;
    t->device = NULL;
    t->ndim = ndim;

    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            tensor_free(t);
            return NULL;
        }
        if (total > SIZE_MAX / (size_t)shape[i]) {
            tensor_free(t);
            return NULL;
        }
        total *= (size_t)shape[i];
    }
    t->size = (int)total;

    if (total > SIZE_MAX / sizeof(float)) {
        tensor_free(t);
        return NULL;
    }

    t->shape = (int *)malloc(ndim * sizeof(int));
    if (t->shape == NULL) {
        tensor_free(t);
        return NULL;
    }
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }

    t->data = (float *)calloc(total, sizeof(float));
    if (t->data == NULL) {
        tensor_free(t);
        return NULL;
    }
    return t;
}

void tensor_free(Tensor *t) {
    if (t == NULL) return;

    if (t->data != NULL) {
        free(t->data);
        t->data = NULL;
    }

    if (t->shape != NULL) {
        free(t->shape);
        t->shape = NULL;
    }

    free(t);
}

Tensor *tensor_random(int ndim, const int *shape, float min, float max) {
    Tensor *t = tensor_create(ndim, shape);

    if (t == NULL) return NULL;

    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }

    for (int i = 0; i < t->size; i++) {
        t->data[i] = min + (float)rand() / RAND_MAX * (max - min);
    }

    return t;
}

Tensor *tensor_from_array(const void *src, DType type, int ndim, const int *shape) {
    if (src == NULL) return NULL;

    Tensor *t = tensor_create(ndim, shape);
    if (t == NULL) return NULL;

    switch (type) {
        case DTYPE_FLOAT: {
            const float *f = (const float *)src;
            for (int i = 0; i < t->size; i++) {
                t->data[i] = f[i];
            }
            break;
        }
        case DTYPE_INT: {
            const int *p = (const int *)src;
            for (int i = 0; i < t->size; i++) {
                t->data[i] = (float)p[i];
            }
            break;
        }
        case DTYPE_BOOL: {
            const int *p = (const int *)src;
            for (int i = 0; i < t->size; i++) {
                t->data[i] = p[i] ? 1.0f : 0.0f;
            }
            break;
        }
        default:
            tensor_free(t);
            return NULL;
    }

    return t;
}

float *tensor_at(const Tensor *t, const int *indices) {
    if (t == NULL || indices == NULL) return NULL;

    int offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) return NULL;
        offset = offset * t->shape[i] + indices[i];
    }
    return &t->data[offset];
}

Tensor *tensor_reshape(const Tensor *t, int ndim, const int *shape) {
    if (t == NULL || shape == NULL || ndim <= 0) return NULL;

    int new_size = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return NULL;
        new_size *= shape[i];
    }

    if (t->size != new_size) return NULL;

    Tensor *new_t = tensor_create(ndim, shape);
    if (new_t == NULL) return NULL;

    memcpy(new_t->data, t->data, t->size * sizeof(float));

    return new_t;
}

Tensor *tensor_transpose(const Tensor *t, int dim0, int dim1) {
    if (t == NULL || dim0 < 0 || dim1 < 0 || dim0 >= t->ndim || dim1 >= t->ndim)
        return NULL;

    if (dim0 == dim1) return tensor_from_array(t->data, DTYPE_FLOAT, t->ndim, t->shape);

    int new_shape[t->ndim];
    memcpy(new_shape, t->shape, t->ndim * sizeof(int));
    int tmp = new_shape[dim0];
    new_shape[dim0] = new_shape[dim1];
    new_shape[dim1] = tmp;

    Tensor *new_t = tensor_create(t->ndim, new_shape);
    if (new_t == NULL) return NULL;

    int orig_strides[t->ndim];
    int stride = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        orig_strides[d] = stride;
        stride *= t->shape[d];
    }

    for (int i = 0; i < new_t->size; i++) {
        int new_idx[t->ndim];
        int rem = i;
        for (int d = t->ndim - 1; d >= 0; d--) {
            new_idx[d] = rem % new_t->shape[d];
            rem /= new_t->shape[d];
        }

        int swp = new_idx[dim0];
        new_idx[dim0] = new_idx[dim1];
        new_idx[dim1] = swp;

        int off = 0;
        for (int d = 0; d < t->ndim; d++) off += new_idx[d] * orig_strides[d];

        new_t->data[i] = t->data[off];
    }

    return new_t;
}
