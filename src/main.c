/**
 * @file main.c
 * @brief Unit tests for the tensor module.
 *
 * Covers creation, random initialization, conversion from C arrays (float, int, bool),
 * and null-parameter handling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ml/core/tensor.h"

/** @brief Tests tensor creation, shape/size correctness, and deallocation. */
static int test_create_and_free(void) {
    int shape[] = {2, 3};
    Tensor *t = tensor_create(2, shape);
    if (t == NULL) return 0;
    if (t->ndim != 2) return 0;
    if (t->size != 6) return 0;
    if (t->shape[0] != 2 || t->shape[1] != 3) return 0;
    tensor_free(t);
    return 1;
}

/** @brief Tests that random values fall within the requested [min, max] range. */
static int test_random(void) {
    int shape[] = {10};
    Tensor *t = tensor_random(1, shape, -1.0f, 1.0f);
    if (t == NULL) return 0;
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] < -1.0f || t->data[i] > 1.0f) return 0;
    }
    tensor_free(t);
    return 1;
}

/** @brief Tests tensor_from_array with DTYPE_FLOAT source data. */
static int test_from_array_float(void) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {2, 2};
    Tensor *t = tensor_from_array(src, DTYPE_FLOAT, 2, shape);
    if (t == NULL) return 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(t->data[i] - src[i]) > 1e-6f) return 0;
    }
    tensor_free(t);
    return 1;
}

/** @brief Tests tensor_from_array with DTYPE_INT source data (int → float cast). */
static int test_from_array_int(void) {
    int src[] = {10, 20};
    int shape[] = {2, 1};
    Tensor *t = tensor_from_array(src, DTYPE_INT, 2, shape);
    if (t == NULL) return 0;
    if (fabsf(t->data[0] - 10.0f) > 1e-6f) return 0;
    if (fabsf(t->data[1] - 20.0f) > 1e-6f) return 0;
    tensor_free(t);
    return 1;
}

/** @brief Tests tensor_from_array with DTYPE_BOOL source data (0/1 → 0.0f/1.0f). */
static int test_from_array_bool(void) {
    int src[] = {1, 0, 1};
    int shape[] = {3};
    Tensor *t = tensor_from_array(src, DTYPE_BOOL, 1, shape);
    if (t == NULL) return 0;
    if (fabsf(t->data[0] - 1.0f) > 1e-6f) return 0;
    if (fabsf(t->data[1] - 0.0f) > 1e-6f) return 0;
    if (fabsf(t->data[2] - 1.0f) > 1e-6f) return 0;
    tensor_free(t);
    return 1;
}

/** @brief Tests that invalid parameters (zero ndim, NULL shape, zero-sized dims)
 *         are rejected with NULL. */
static int test_null_params(void) {
    Tensor *t = tensor_create(0, NULL);
    if (t != NULL) return 0;
    t = tensor_create(1, NULL);
    if (t != NULL) return 0;
    int shape[] = {0};
    t = tensor_create(1, shape);
    if (t != NULL) return 0;
    return 1;
}

/**
 * @brief Runs all tensor unit tests and reports pass/fail counts.
 *
 * @return 0 on full pass, 1 if any test failed.
 */
int main(void) {
    int passed = 0, failed = 0;

    printf("=== Tensor tests ===\n");

    if (test_create_and_free())  { printf("[PASS] create_and_free\n");  passed++; }
    else                         { printf("[FAIL] create_and_free\n");  failed++; }

    if (test_random())           { printf("[PASS] random\n");           passed++; }
    else                         { printf("[FAIL] random\n");           failed++; }

    if (test_from_array_float())  { printf("[PASS] from_array_float\n");  passed++; }
    else                          { printf("[FAIL] from_array_float\n");  failed++; }

    if (test_from_array_int())    { printf("[PASS] from_array_int\n");    passed++; }
    else                          { printf("[FAIL] from_array_int\n");    failed++; }

    if (test_from_array_bool())   { printf("[PASS] from_array_bool\n");   passed++; }
    else                          { printf("[FAIL] from_array_bool\n");   failed++; }

    if (test_null_params())      { printf("[PASS] null_params\n");      passed++; }
    else                         { printf("[FAIL] null_params\n");      failed++; }

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
