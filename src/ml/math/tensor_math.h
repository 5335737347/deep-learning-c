// tensor_math.h — element-wise tensor operations with NumPy-style broadcasting.

#ifndef SRC_ML_TENSOR_MATH_H_
#define SRC_ML_TENSOR_MATH_H_

#include "ml/core/tensor.h"
#include <stddef.h>

Tensor *tensor_add(const Tensor *input, const Tensor *other);

Tensor *tensor_sub(const Tensor *input, const Tensor *other);

Tensor *tensor_mul(const Tensor *input, const Tensor *other);

Tensor *tensor_div(const Tensor *input, const Tensor *other);

Tensor *tensor_sum(const Tensor *input, size_t dim);

Tensor *tensor_mean(const Tensor *input, size_t dim);
#endif
