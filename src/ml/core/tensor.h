/**
 * @file tensor.h
 * @brief N-dimensional tensor data structure — the core numeric container for all ML
 *        operations.
 */

#ifndef SRC_ML_TENSOR_H_
#define SRC_ML_TENSOR_H_

#include "ml/device/device.h"

/**
 * @brief N-dimensional array (tensor) backed by a contiguous float buffer.
 *
 * All numerical computations in the library operate on Tensors. The underlying data
 * is stored in row-major order.
 */
typedef struct Tensor {
    float *data;       /**< Flattened data buffer in row-major order. */
    int *shape;        /**< Array of per-dimension sizes (length = ndim). */
    int ndim;          /**< Number of dimensions (rank). */
    int size;          /**< Total number of elements (product of shape). */
    struct Device *device; /**< Compute device this tensor resides on (or NULL for CPU). */
} Tensor;

/** @brief Data type of source elements for tensor_from_array(). */
typedef enum {
    DTYPE_FLOAT,  /**< float values. */
    DTYPE_INT,    /**< int values (cast to float). */
    DTYPE_BOOL,   /**< 0/1 int values (mapped to 0.0f/1.0f). */
} DType;

/**
 * @brief Allocates and returns a zero-initialized tensor with the given shape.
 *
 * @param ndim  Number of dimensions (must be > 0).
 * @param shape Array of dimension sizes (each must be > 0).
 * @return Pointer to the allocated Tensor, or NULL on failure (invalid args or OOM).
 */
Tensor *tensor_create(int ndim, const int *shape);

/**
 * @brief Allocates a tensor and fills it with uniformly-distributed random values.
 *
 * @param ndim  Number of dimensions.
 * @param shape Array of dimension sizes.
 * @param min   Lower bound (inclusive) of the uniform distribution.
 * @param max   Upper bound (inclusive) of the uniform distribution.
 * @return Pointer to the allocated Tensor, or NULL on failure.
 */
Tensor *tensor_random(int ndim, const int *shape, float min, float max);

/**
 * @brief Creates a tensor by copying data from a C array.
 *
 * Accepts float, int, or bool source arrays and converts them to the internal float
 * representation.
 *
 * @param src   Source array pointer.
 * @param type  Element type of the source array (one of DTYPE_*).
 * @param ndim  Number of dimensions.
 * @param shape Array of dimension sizes.
 * @return Pointer to the allocated Tensor, or NULL on failure.
 */
Tensor *tensor_from_array(const void *src, DType type, int ndim, const int *shape);

/**
 * @brief Returns a new tensor with the same data reshaped to the given shape.
 *
 * The total number of elements must match the source tensor.
 *
 * @param t     Source tensor.
 * @param ndim  Number of dimensions of the new shape.
 * @param shape New shape array.
 * @return Pointer to the new Tensor, or NULL on failure.
 */
Tensor *tensor_reshape(const Tensor *t, int ndim, const int *shape);

/**
 * @brief Returns a new tensor with two dimensions swapped.
 *
 * @param t    Source tensor.
 * @param dim0 First dimension to swap.
 * @param dim1 Second dimension to swap.
 * @return Pointer to the new transposed Tensor, or NULL on failure.
 */
Tensor *tensor_transpose(const Tensor *t, int dim0, int dim1);

/**
 * @brief Returns a pointer to the element at the given multi-dimensional index.
 *
 * @param t       Tensor to index into.
 * @param indices Array of integer indices (length = t->ndim).
 * @return Pointer to the element, or NULL if indices are out of range.
 */
float *tensor_at(const Tensor *t, const int *indices);

/**
 * @brief Frees all memory associated with the tensor.
 *
 * @param t Tensor to free (NULL is safely ignored).
 */
void tensor_free(Tensor *t);

#endif
