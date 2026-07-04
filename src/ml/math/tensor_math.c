#include <stdlib.h>
#include <threads.h>

#include "tensor_math.h"

static float add_op(float a, float b) { return a + b; }
static float sub_op(float a, float b) { return a - b; }
static float mul_op(float a, float b) { return a * b; }
static float div_op(float a, float b) { return a / b; }

static Tensor *tensor_broadcast_binary(const Tensor *a, const Tensor *b,
                                        float (*op)(float, float)) {
    if (a == NULL || b == NULL) return NULL;

    size_t out_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    size_t *out_shape = (size_t *)malloc(out_ndim * sizeof(size_t));
    if (out_shape == NULL) return NULL;

    for (int i = 0; i < out_ndim; i++) {
        int rev = out_ndim - 1 - i;
        int sz_a = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
        int sz_b = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;
        if (sz_a != sz_b && sz_a != 1 && sz_b != 1) { free(out_shape); return NULL; }
        out_shape[rev] = (size_t)(sz_a > sz_b ? sz_a : sz_b);
    }

    Tensor *result = tensor_create(out_ndim, out_shape);
    free(out_shape);
    if (result == NULL) return NULL;

    int a_strides[a->ndim];
    int stride = 1;
    for (int d = a->ndim - 1; d >= 0; d--) {
        a_strides[d] = stride;
        stride *= a->shape[d];
    }

    int b_strides[b->ndim];
    stride = 1;
    for (int d = b->ndim - 1; d >= 0; d--) {
        b_strides[d] = stride;
        stride *= b->shape[d];
    }

    int out_strides[out_ndim];
    stride = 1;
    for (int d = out_ndim - 1; d >= 0; d--) {
        out_strides[d] = stride;
        stride *= out_shape[d];
    }

    for (int i = 0; i < result->size; i++) {
        int multi[out_ndim];
        int rem = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            multi[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        int a_flat = 0;
        for (int d = 0; d < out_ndim; d++) {
            int ad = d - (out_ndim - a->ndim);
            if (ad >= 0) {
                int idx = a->shape[ad] == 1 ? 0 : multi[d];
                a_flat += idx * a_strides[ad];
            }
        }

        int b_flat = 0;
        for (int d = 0; d < out_ndim; d++) {
            int bd = d - (out_ndim - b->ndim);
            if (bd >= 0) {
                int idx = b->shape[bd] == 1 ? 0 : multi[d];
                b_flat += idx * b_strides[bd];
            }
        }

        result->storage->data[i] = op(a->storage->data[a_flat],
                                       b->storage->data[b_flat]);
    }

    return result;
}

Tensor *tensor_add(const Tensor *input, const Tensor *other) {
    return tensor_broadcast_binary(input, other, add_op);
}

Tensor *tensor_sub(const Tensor *input, const Tensor *other) {
    return tensor_broadcast_binary(input, other, sub_op);
}

Tensor *tensor_mul(const Tensor *input, const Tensor *other) {
    return tensor_broadcast_binary(input, other, mul_op);
}

Tensor *tensor_div(const Tensor *input, const Tensor *other) {
    return tensor_broadcast_binary(input, other, div_op);
}
