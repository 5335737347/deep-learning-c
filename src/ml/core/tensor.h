// tensor.h — N-dimensional tensor data structure, the core numeric container for
// all ML operations.
//
// Dimensions, shapes, and strides use int64_t to match PyTorch / NumPy conventions
// and to support negative strides for zero-copy flip / reverse operations.

#ifndef SRC_ML_TENSOR_H_
#define SRC_ML_TENSOR_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ml/device/device.h"

typedef enum {
    DEVICE_CPU = 0,
    DEVICE_CUDA,
} DeviceType;

typedef struct TensorStorage {
    float *data;
    size_t size;       /* element count — kept as size_t for malloc compatibility */
    int ref_count;
    DeviceType device;
} Storage;

Storage *storage_create(size_t n, bool zero_init, DeviceType device);

void storage_retain(Storage *s);

void storage_release(Storage *s);

// N-dimensional array (tensor) backed by a contiguous float buffer.
//
// All numerical computations in the library operate on Tensors. The underlying data
// is stored in row-major order. Strides may be negative to represent flipped axes.
typedef struct {
    Storage *storage;
    int64_t *shape;
    int64_t *strides;   /* signed — enables zero-copy flip */
    int64_t offset;
    int64_t ndim;
    int64_t size;       /* logical element count (≤ INT64_MAX) */
} Tensor;

// Allocates and returns a zero-initialized tensor with the given shape.
//
// ndim:  Number of dimensions (must be > 0).
// shape: Array of dimension sizes (each must be > 0).
// Returns: Pointer to the allocated Tensor, or NULL on failure.
Tensor *tensor_create(int64_t ndim, const int64_t *shape);

Tensor *tensor_create_view(Storage *storage, int64_t ndim,
                           const int64_t *shape, const int64_t *strides,
                           int64_t offset);

void tensor_free(Tensor *t);

int64_t tensor_flat_index(const Tensor *t, const int64_t *indices);

float *tensor_at(const Tensor *t, const int64_t *indices);

bool tensor_is_contiguous(const Tensor *t);

Tensor *tensor_contiguous(const Tensor *t);

Tensor *tensor_transpose(const Tensor *t, int64_t dim0, int64_t dim1);

Tensor *tensor_slice(const Tensor *t, int64_t dim, int64_t start, int64_t end);

Tensor *tensor_reshape(const Tensor *t, int64_t new_ndim, const int64_t *new_shape);

Tensor *tensor_broadcast_to(const Tensor *t, int64_t target_ndim,
                            const int64_t *target_shape);

Tensor *tensor_squeeze(const Tensor *t);

Tensor *tensor_unsqueeze(const Tensor *t, int64_t dim);

// Returns a deep copy with its own storage.
//
// The result is always contiguous, regardless of whether the source was.
Tensor *tensor_clone(const Tensor *t);

// Sets every element of the tensor to the given value.
void tensor_fill(Tensor *t, float value);

void tensor_print(const Tensor *t);

#endif
