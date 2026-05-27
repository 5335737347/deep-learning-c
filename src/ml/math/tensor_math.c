#include <stdlib.h>

#include "tensor_math.h"
#include "ml/core/tensor.h"

Tensor *tensor_add(const Tensor *a, const Tensor *b) {
    if (a == NULL || b == NULL) return NULL;

    if (a->ndim != b->ndim) return NULL;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }

    Tensor *result = tensor_create(a->ndim, a->shape);
    if (result == NULL) return NULL;

    for (int i = 0; i < a->size; i++) result->data[i] = a->data[i] + b->data[i];

    return result;
}
