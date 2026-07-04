#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ml/core/tensor.h"

/* ─── helpers ────────────────────────────────────────────────────────── */

static Tensor *tensor_from_flat(const float *data, int64_t ndim,
                                const int64_t *shape) {
    Tensor *t = tensor_create(ndim, shape);
    if (!t) return NULL;
    for (int64_t i = 0; i < t->size; i++)
        t->storage->data[i] = data[i];
    return t;
}

static int pass = 0, fail = 0;

#define TEST(name)                                                \
    do {                                                          \
        if (test_##name()) { ++pass; printf("[PASS] " #name "\n"); }  \
        else              { ++fail; printf("[FAIL] " #name "\n"); }  \
    } while (0)

/* ─── test cases ─────────────────────────────────────────────────────── */

static int test_create_basic(void) {
    int64_t sh[] = {2, 3};
    Tensor *t = tensor_create(2, sh);
    if (!t) return 0;
    int ok = t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3 &&
             t->size == 6 && t->offset == 0 &&
             t->strides[0] == 3 && t->strides[1] == 1 &&
             tensor_is_contiguous(t);
    /* values are zero-initialized */
    for (int64_t i = 0; ok && i < 6; i++)
        if (t->storage->data[i] != 0.0f) ok = 0;
    tensor_free(t);
    return ok;
}

static int test_create_scalar_ndim1(void) {
    int64_t sh[] = {1};
    Tensor *t = tensor_create(1, sh);
    if (!t) return 0;
    int ok = t->ndim == 1 && t->shape[0] == 1 && t->size == 1 &&
             t->strides[0] == 1;
    tensor_free(t);
    return ok;
}

static int test_create_invalid_ndim(void) {
    int64_t sh[] = {2, 3};
    Tensor *t = tensor_create(0, sh);
    return t == NULL && errno == EINVAL;
}

static int test_create_null_shape(void) {
    Tensor *t = tensor_create(2, NULL);
    return t == NULL && errno == EINVAL;
}

static int test_create_zero_dim(void) {
    int64_t sh[] = {2, 0, 3};
    Tensor *t = tensor_create(3, sh);
    return t == NULL && errno == EINVAL;
}

static int test_at_and_flat_index(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    int ok = 1;
    /* tensor_at */
    int64_t idx00[] = {0, 0}, idx01[] = {0, 1}, idx12[] = {1, 2};
    if (*tensor_at(t, idx00) != 1.0f) ok = 0;
    if (*tensor_at(t, idx01) != 2.0f) ok = 0;
    if (*tensor_at(t, idx12) != 6.0f) ok = 0;

    /* tensor_flat_index should match linear offset */
    if (tensor_flat_index(t, idx00) != 0) ok = 0;
    if (tensor_flat_index(t, idx01) != 1) ok = 0;
    if (tensor_flat_index(t, idx12) != 5) ok = 0;

    /* out-of-bounds */
    int64_t oob[] = {2, 0};
    if (tensor_at(t, oob) != NULL) ok = 0;

    /* NULL safety */
    if (tensor_at(NULL, idx00) != NULL) ok = 0;
    if (tensor_flat_index(NULL, idx00) != 0) ok = 0;

    tensor_free(t);
    return ok;
}

static int test_create_view(void) {
    int64_t sh[] = {3};
    float data[] = {10, 20, 30};
    Tensor *orig = tensor_from_flat(data, 1, sh);
    if (!orig) return 0;

    /* view of the whole tensor */
    int64_t strides[] = {1};
    Tensor *view = tensor_create_view(orig->storage, 1, sh, strides, 0);
    if (!view) { tensor_free(orig); return 0; }

    int64_t idx0[] = {0}, idx2[] = {2};
    int ok = (*tensor_at(view, idx0) == 10.0f) &&
             (*tensor_at(view, idx2) == 30.0f) &&
             view->storage == orig->storage;  /* shared storage */

    tensor_free(view);
    tensor_free(orig);
    return ok;
}

static int test_view_bounds_check(void) {
    int64_t sh[] = {3};
    float data[] = {10, 20, 30};
    Tensor *orig = tensor_from_flat(data, 1, sh);
    if (!orig) return 0;

    /* pretend stride is 2, starting at offset 0:
       last element = 0 + 2*(3-1) = 4 >= storage->size (3) */
    int64_t bad_strides[] = {2};
    Tensor *view = tensor_create_view(orig->storage, 1, sh, bad_strides, 0);
    int ok = (view == NULL);

    tensor_free(view);
    tensor_free(orig);
    return ok;
}

static int test_is_contiguous(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;
    int ok = tensor_is_contiguous(t);
    tensor_free(t);
    return ok && !tensor_is_contiguous(NULL);
}

static int test_transpose(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    Tensor *tt = tensor_transpose(t, 0, 1);
    if (!tt) { tensor_free(t); return 0; }

    int ok = tt->ndim == 2 && tt->shape[0] == 3 && tt->shape[1] == 2 &&
             !tensor_is_contiguous(tt);

    /* element (0,1) of tt should be data at orig (1,0) = 4 */
    int64_t idx[] = {0, 1};
    ok = ok && (*tensor_at(tt, idx) == 4.0f);

    /* invalid dims */
    Tensor *bad = tensor_transpose(t, 0, 5);
    ok = ok && (bad == NULL);

    tensor_free(tt);
    tensor_free(t);
    return ok;
}

static int test_slice(void) {
    int64_t sh[] = {3, 4};
    float data[] = {0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    /* slice dim=0, rows [1:3) → 2x4 tensor with rows {4,5,6,7} and {8,9,10,11} */
    Tensor *s = tensor_slice(t, 0, 1, 3);
    if (!s) { tensor_free(t); return 0; }

    int ok = s->ndim == 2 && s->shape[0] == 2 && s->shape[1] == 4;
    int64_t idx00[] = {0, 0}, idx12[] = {1, 2};
    ok = ok && (*tensor_at(s, idx00) == 4.0f);   /* row 1, col 0 */
    ok = ok && (*tensor_at(s, idx12) == 10.0f);  /* row 2, col 2 */

    /* invalid slice */
    Tensor *bad = tensor_slice(t, 0, 2, 1);
    ok = ok && (bad == NULL);

    tensor_free(s);
    tensor_free(t);
    return ok;
}

static int test_reshape_contiguous(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    int64_t new_sh[] = {3, 2};
    Tensor *r = tensor_reshape(t, 2, new_sh);
    if (!r) { tensor_free(t); return 0; }

    int ok = r->ndim == 2 && r->shape[0] == 3 && r->shape[1] == 2 &&
             r->size == 6;
    int64_t idx[] = {0, 1};
    ok = ok && (*tensor_at(r, idx) == 2.0f);

    tensor_free(r);
    tensor_free(t);
    return ok;
}

static int test_reshape_non_contiguous(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    /* transpose makes it non-contiguous, then reshape to 1-d */
    Tensor *tt = tensor_transpose(t, 0, 1);  /* 3x2 */
    if (!tt) { tensor_free(t); return 0; }

    int64_t new_sh[] = {6};
    Tensor *r = tensor_reshape(tt, 1, new_sh);
    if (!r) { tensor_free(tt); tensor_free(t); return 0; }

    /* After contiguous-ify & reshape: should be [1, 4, 2, 5, 3, 6] */
    float expected[] = {1, 4, 2, 5, 3, 6};
    int ok = r->size == 6 && r->ndim == 1 && r->shape[0] == 6;
    for (int64_t i = 0; ok && i < 6; i++)
        if (fabsf(r->storage->data[i] - expected[i]) > 1e-6f) ok = 0;

    tensor_free(r);
    tensor_free(tt);
    tensor_free(t);
    return ok;
}

static int test_reshape_size_mismatch(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    int64_t bad_sh[] = {7};  /* 7 != 6 */
    Tensor *r = tensor_reshape(t, 1, bad_sh);
    int ok = (r == NULL);

    tensor_free(t);
    return ok;
}

static int test_broadcast_to(void) {
    int64_t sh[] = {3};
    float data[] = {1, 2, 3};
    Tensor *t = tensor_from_flat(data, 1, sh);
    if (!t) return 0;

    int64_t target[] = {2, 3};
    Tensor *b = tensor_broadcast_to(t, 2, target);
    if (!b) { tensor_free(t); return 0; }

    int ok = b->ndim == 2 && b->shape[0] == 2 && b->shape[1] == 3 &&
             b->size == 6 && b->strides[0] == 0 && b->strides[1] == 1;

    /* row 0 and row 1 should be identical */
    int64_t r0[] = {0, 1}, r1[] = {1, 1};
    ok = ok && (*tensor_at(b, r0) == 2.0f) && (*tensor_at(b, r1) == 2.0f);

    /* incompatible: ndim too large */
    Tensor *bad = tensor_broadcast_to(t, 1, target);
    ok = ok && (bad == NULL);

    tensor_free(b);
    tensor_free(t);
    return ok;
}

static int test_broadcast_scalar(void) {
    int64_t sh[] = {1};
    float data[] = {5.0f};
    Tensor *t = tensor_from_flat(data, 1, sh);
    if (!t) return 0;

    int64_t target[] = {3, 4};
    Tensor *b = tensor_broadcast_to(t, 2, target);
    if (!b) { tensor_free(t); return 0; }

    int ok = b->ndim == 2 && b->shape[0] == 3 && b->shape[1] == 4;
    int64_t idx[] = {2, 3};
    ok = ok && (*tensor_at(b, idx) == 5.0f);

    tensor_free(b);
    tensor_free(t);
    return ok;
}

static int test_broadcast_incompatible(void) {
    int64_t sh[] = {2};
    float data[] = {1, 2};
    Tensor *t = tensor_from_flat(data, 1, sh);
    if (!t) return 0;

    int64_t target[] = {3};  /* 2 != 3 and 2 != 1 */
    Tensor *b = tensor_broadcast_to(t, 1, target);
    int ok = (b == NULL);

    tensor_free(t);
    return ok;
}

static int test_squeeze(void) {
    int64_t sh[] = {1, 3, 1, 2};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 4, sh);
    if (!t) return 0;

    Tensor *sq = tensor_squeeze(t);
    if (!sq) { tensor_free(t); return 0; }

    int ok = sq->ndim == 2 && sq->shape[0] == 3 && sq->shape[1] == 2 &&
             sq->size == 6;
    /* values preserved */
    int64_t idx[] = {1, 0};
    ok = ok && (*tensor_at(sq, idx) == 3.0f);

    tensor_free(sq);

    /* no size-1 dims: squeeze returns same-ndim view */
    int64_t sh2[] = {2, 3};
    Tensor *t2 = tensor_from_flat(data, 2, sh2);
    Tensor *sq2 = tensor_squeeze(t2);
    ok = ok && sq2->ndim == 2;
    tensor_free(sq2);
    tensor_free(t2);
    tensor_free(t);

    return ok;
}

static int test_unsqueeze(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    /* insert dim at position 0 */
    Tensor *u0 = tensor_unsqueeze(t, 0);
    int ok = u0 && u0->ndim == 3 && u0->shape[0] == 1 && u0->shape[1] == 2 &&
             u0->shape[2] == 3;
    tensor_free(u0);

    /* insert dim at end */
    Tensor *u2 = tensor_unsqueeze(t, 2);
    ok = ok && u2 && u2->ndim == 3 && u2->shape[0] == 2 && u2->shape[1] == 3 &&
         u2->shape[2] == 1;
    tensor_free(u2);

    /* invalid dim */
    Tensor *bad = tensor_unsqueeze(t, 5);
    ok = ok && (bad == NULL);

    tensor_free(t);
    return ok;
}

static int test_contiguous(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    /* already contiguous → should get a view, not a copy (same storage) */
    Tensor *c1 = tensor_contiguous(t);
    int ok = c1 && c1->storage == t->storage;
    tensor_free(c1);

    /* transposed → non-contiguous → contiguous should make a copy */
    Tensor *tt = tensor_transpose(t, 0, 1);
    Tensor *c2 = tensor_contiguous(tt);
    /* after contiguous, data order should match row-major of the transposed view:
       tt is 3x2: [[1,4],[2,5],[3,6]] */
    float expected[] = {1, 4, 2, 5, 3, 6};
    ok = ok && c2 && c2->storage != tt->storage && tensor_is_contiguous(c2);
    for (int64_t i = 0; ok && i < 6; i++)
        if (fabsf(c2->storage->data[i] - expected[i]) > 1e-6f) ok = 0;

    tensor_free(c2);
    tensor_free(tt);
    tensor_free(t);
    return ok;
}

static int test_ref_count(void) {
    int64_t sh[] = {4};
    float data[] = {1, 2, 3, 4};
    Tensor *t = tensor_from_flat(data, 1, sh);
    if (!t) return 0;

    int rc0 = t->storage->ref_count;  /* should be 1 */

    int64_t strides[] = {1};
    Tensor *v1 = tensor_create_view(t->storage, 1, sh, strides, 0);
    Tensor *v2 = tensor_create_view(t->storage, 1, sh, strides, 0);

    int rc1 = t->storage->ref_count;  /* should be 3 */
    int ok = (rc0 == 1) && (rc1 == 3);

    /* free views; original tensor should still be valid */
    tensor_free(v1);
    tensor_free(v2);
    ok = ok && (t->storage->ref_count == 1);
    int64_t idx[] = {2};
    ok = ok && (*tensor_at(t, idx) == 3.0f);

    tensor_free(t);
    return ok;
}

static int test_errno_cleared(void) {
    /* errno should not be set on success paths */
    errno = 0;
    int64_t sh[] = {2, 2};
    Tensor *t = tensor_create(2, sh);
    int ok = (t != NULL) && (errno == 0);
    tensor_free(t);
    return ok;
}

static int test_clone(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    Tensor *c = tensor_clone(t);
    if (!c) { tensor_free(t); return 0; }

    /* clone must be contiguous */
    int ok = tensor_is_contiguous(c);
    /* clone must have its own storage */
    ok = ok && (c->storage != t->storage);
    /* values must match */
    for (int64_t i = 0; ok && i < 6; i++)
        if (fabsf(c->storage->data[i] - data[i]) > 1e-6f) ok = 0;
    /* original tensor must be unaffected */
    ok = ok && (t->storage->ref_count == 1);
    /* NULL safety */
    ok = ok && (tensor_clone(NULL) == NULL);

    tensor_free(c);
    tensor_free(t);
    return ok;
}

static int test_clone_non_contiguous(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    Tensor *tt = tensor_transpose(t, 0, 1);  /* 3x2, non-contiguous */
    if (!tt) { tensor_free(t); return 0; }

    Tensor *c = tensor_clone(tt);
    if (!c) { tensor_free(tt); tensor_free(t); return 0; }

    /* clone of non-contiguous must be contiguous */
    int ok = tensor_is_contiguous(c);
    /* values should be in row-major order of the transposed view */
    float expected[] = {1, 4, 2, 5, 3, 6};
    for (int64_t i = 0; ok && i < 6; i++)
        if (fabsf(c->storage->data[i] - expected[i]) > 1e-6f) ok = 0;

    tensor_free(c);
    tensor_free(tt);
    tensor_free(t);
    return ok;
}

static int test_fill_contiguous(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    tensor_fill(t, 7.0f);
    int64_t idx[] = {0, 1};
    int ok = (*tensor_at(t, idx) == 7.0f);
    /* all elements should be 7.0 */
    for (int64_t i = 0; ok && i < 6; i++)
        if (fabsf(t->storage->data[i] - 7.0f) > 1e-6f) ok = 0;

    /* NULL safety (must not crash) */
    tensor_fill(NULL, 0.0f);

    tensor_free(t);
    return ok;
}

static int test_fill_broadcasted(void) {
    int64_t sh[] = {3};
    float data[] = {1, 2, 3};
    Tensor *t = tensor_from_flat(data, 1, sh);
    if (!t) return 0;

    /* broadcast to 2x3 → non-contiguous (stride-0 on dim 0) */
    int64_t target[] = {2, 3};
    Tensor *b = tensor_broadcast_to(t, 2, target);
    if (!b) { tensor_free(t); return 0; }

    /* fill should touch underlying storage via all elements */
    tensor_fill(b, 42.0f);
    int ok = (t->storage->data[0] == 42.0f &&
              t->storage->data[1] == 42.0f &&
              t->storage->data[2] == 42.0f);

    tensor_free(b);
    tensor_free(t);
    return ok;
}

static int test_transpose_noop(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    /* transpose same dims → should share storage (view, not copy) */
    Tensor *tt = tensor_transpose(t, 0, 0);
    int ok = tt && tt->storage == t->storage && tt->ndim == 2;
    ok = ok && tt->shape[0] == 2 && tt->shape[1] == 3;

    tensor_free(tt);
    tensor_free(t);
    return ok;
}

static int test_negative_dim_guard(void) {
    int64_t sh[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor *t = tensor_from_flat(data, 2, sh);
    if (!t) return 0;

    /* negative dims should be rejected */
    int ok = (tensor_transpose(t, -1, 0) == NULL);
    ok = ok && (tensor_slice(t, -1, 0, 1) == NULL);

    tensor_free(t);
    return ok;
}

/* ─── main ───────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== tensor core tests ===\n\n");

    TEST(create_basic);
    TEST(create_scalar_ndim1);
    TEST(create_invalid_ndim);
    TEST(create_null_shape);
    TEST(create_zero_dim);
    TEST(at_and_flat_index);
    TEST(create_view);
    TEST(view_bounds_check);
    TEST(is_contiguous);
    TEST(transpose);
    TEST(slice);
    TEST(reshape_contiguous);
    TEST(reshape_non_contiguous);
    TEST(reshape_size_mismatch);
    TEST(broadcast_to);
    TEST(broadcast_scalar);
    TEST(broadcast_incompatible);
    TEST(squeeze);
    TEST(unsqueeze);
    TEST(contiguous);
    TEST(ref_count);
    TEST(errno_cleared);
    TEST(clone);
    TEST(clone_non_contiguous);
    TEST(fill_contiguous);
    TEST(fill_broadcasted);
    TEST(transpose_noop);
    TEST(negative_dim_guard);

    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
