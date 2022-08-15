import numpy as np

import scipy
import scipy.sparse as sparse
import scipy.linalg as linalg

from .utils import *
from .mkl_funcs import *
from .utils cimport *
from .mkl_funcs cimport *

cimport numpy as np


cpdef float[:] sparse_mat_rowsum(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        bint sptype
        ):

    cdef float[:] ones = np.ones(ncol, dtype=np.float32)
    cdef float[:] rowsum = mkl_sparse_mv(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            sptype,
            ones,
            0
            )

    return rowsum

cpdef float[:] sparse_mat_colsum(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        bint sptype
        ):

    cdef float[:] ones = np.ones(nrow, dtype=np.float32)
    cdef float[:] colsum = mkl_sparse_mv(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            sptype,
            ones,
            1
            )

    return colsum


cpdef np.ndarray[np.float32_t, ndim=1] fastrna_hvg_sub(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol
        ):

    # calculate row and column sums
    cdef float[:] n_umi_row = sparse_mat_rowsum(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            0
            )
    cdef float[:] n_umi_col = sparse_mat_colsum(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            0
            )
    cdef float[:] prop_per_cell = norm1(n_umi_col)

    # calculate common components
    cdef float[:] one_sub_prop = pcomp(prop_per_cell)
    cdef float[:] var_prop = mul(prop_per_cell, one_sub_prop)
    cdef float[:] prop_div = div(prop_per_cell, one_sub_prop)

    # calculate first component
    cdef float[:] first_reduce = mkl_sparse_mv(
            csc_div_vec_row(mul(data, data), indices, indptr, n_umi_row),
            indices,
            indptr,
            nrow,
            ncol,
            0,
            inv(var_prop),
            0
            )

    # calculate seoncd component
    cdef float[:] sec_reduce = mkl_sparse_mv(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            0,
            inv(one_sub_prop),
            0
            )

    # calculate thrid component
    cdef float[:] third_reduce = mul_s(n_umi_row, vsum(prop_div))

    return np.asarray(first_reduce) - 2 * np.asarray(sec_reduce) + np.asarray(third_reduce)

cpdef np.ndarray[np.float32_t, ndim=2] fastrna_proj(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        const float[:] n_umi_col,
        np.ndarray eig_vec
        ):

    # calculate common components
    cdef float[:] n_umi_row = sparse_mat_rowsum(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            0)
    cdef float[:] prop_per_cell = norm1(n_umi_col)
    cdef float[:] one_sub_prop = pcomp(prop_per_cell)

    # sqrts
    cdef float[:] n_umi_row_sqrt = sqrt_c(n_umi_row)
    cdef float[:] prop_per_cell_sqrt = sqrt_c(prop_per_cell)
    cdef float[:] one_sub_prop_sqrt = sqrt_c(one_sub_prop)

    # calculate common components
    cdef float[:] data_A_row = csc_div_vec_row(
            data,
            indices,
            indptr,
            n_umi_row_sqrt
           )
    cdef float[:] data_A = csc_div_vec_col(
            data_A_row,
            indices,
            indptr,
            mul(prop_per_cell_sqrt, one_sub_prop_sqrt)
            )

    # calculate first component
    first_csc = sparse.csr_matrix(
            (data_A, indices, indptr),
            (ncol, nrow)
            )
    first_reduce = first_csc.dot(eig_vec)

    # second component
    sec_one = div(prop_per_cell_sqrt, one_sub_prop_sqrt)
    sec_two = np.asarray(n_umi_row_sqrt) @ eig_vec

    return first_reduce - cblas_ger(sec_one, sec_two)

cpdef np.ndarray[np.float32_t, ndim=2] fastrna_ed(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        const float[:] n_umi_col
        ):

    # calculate common components
    cdef float[:] n_umi_row = sparse_mat_rowsum(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            0)
    cdef float[:] prop_per_cell = norm1(n_umi_col)
    cdef float[:] one_sub_prop = pcomp(prop_per_cell)

    # sqrts
    cdef float[:] n_umi_row_sqrt = sqrt_c(n_umi_row)
    cdef float[:] prop_per_cell_sqrt = sqrt_c(prop_per_cell)
    cdef float[:] one_sub_prop_sqrt = sqrt_c(one_sub_prop)

    # calculate common components
    cdef float[:] data_A_row = csc_div_vec_row(
            data,
            indices,
            indptr,
            n_umi_row_sqrt
           )
    cdef float[:] data_A = csc_div_vec_col(
            data_A_row,
            indices,
            indptr,
            mul(prop_per_cell_sqrt, one_sub_prop_sqrt)
            )

    # calculate first component
    cdef np.ndarray[np.float32_t, ndim=2] cov_mat_first = mkl_sparse_gram(
            data_A,
            indices,
            indptr,
            ncol,
            nrow,
            1
            )

    # calculate second component
    # 안 될 땐 input 길이, nrow, ncol, transpose 바뀐 거 먼저 봐라 
    cdef float[:] cov_mat_sec_row = mkl_sparse_mv(
            data_A,
            indices,
            indptr,
            nrow,
            ncol,
            0,
            div(prop_per_cell_sqrt, one_sub_prop_sqrt),
            0
            )
    cdef np.ndarray[np.float32_t, ndim=2] cov_mat_sec = cblas_ger(
            n_umi_row_sqrt, 
            cov_mat_sec_row
            )

    # calculate third component
    cdef float[:] prop_ratio = div(prop_per_cell, one_sub_prop)
    cdef np.ndarray[np.float32_t, ndim=2] cov_mat_third = vsum(prop_ratio) * cblas_ger(
            n_umi_row_sqrt,
            n_umi_row_sqrt
            )

    return cov_mat_first - 2 * cov_mat_sec + cov_mat_third

def fastrna_hvg(
        mtx,
        batch_label=None
        ):

    if batch_label is None:
        batch_label = np.zeros(mtx.shape[1])

    blab_indptr = np.insert(
            np.cumsum(
                np.unique(
                    batch_label,
                    return_counts=True
                    )[1]
                ),
            0,
            0
            )
    expr2 = np.zeros(mtx.shape[0], dtype=np.float32)
    for b in range(len(blab_indptr)-1):
        # sparse indexing
        c_begin = blab_indptr[b]
        c_end = blab_indptr[b+1]

        b_begin = mtx.indptr[c_begin]
        b_end = mtx.indptr[c_end]

        b_data = mtx.data[b_begin:b_end]
        b_indices = mtx.indices[b_begin:b_end]
        b_indptr = mtx.indptr[c_begin:c_end+1] - b_begin

        expr2 += fastrna_hvg_sub(
                b_data,
                b_indices,
                b_indptr,
                mtx.shape[0],
                c_end-c_begin
                )

    return expr2 / mtx.shape[1]


def fastrna_pca(
        mtx,
        n_umi_col,
        batch_label=None,
        k=50
        ):

    if batch_label is None:
        batch_label = np.zeros(mtx.shape[1], dtype=int)

    blab_indptr = np.insert(
            np.cumsum(
                np.unique(
                    batch_label,
                    return_counts=True
                    )[1]
                ),
            0,
            0
            )
    rrt = np.zeros((mtx.shape[0], mtx.shape[0]), dtype=np.float32)
    for b in range(len(blab_indptr)-1):
        # sparse indexing
        c_begin = blab_indptr[b]
        c_end = blab_indptr[b+1]

        b_begin = mtx.indptr[c_begin]
        b_end = mtx.indptr[c_end]

        b_data = mtx.data[b_begin:b_end]
        b_indices = mtx.indices[b_begin:b_end]
        b_indptr = mtx.indptr[c_begin:c_end+1] - b_begin
        nrow, ncol = mtx.shape[0], c_end-c_begin

        b_n_umi_col = n_umi_col[c_begin:c_end]
        rrt += fastrna_ed(
                b_data,
                b_indices,
                b_indptr,
                nrow,
                ncol,
                b_n_umi_col
                )

    eig_val, eig_vec = linalg.eigh(
            a=rrt,
            lower=False,
            subset_by_index=[mtx.shape[0]-k, mtx.shape[0]-1]
            )

    pca_coord = np.zeros((mtx.shape[1], k), dtype=np.float32)
    for b in range(len(blab_indptr)-1):
        # sparse indexing
        c_begin = blab_indptr[b]
        c_end = blab_indptr[b+1]

        b_begin = mtx.indptr[c_begin]
        b_end = mtx.indptr[c_end]

        b_data = mtx.data[b_begin:b_end]
        b_indices = mtx.indices[b_begin:b_end]
        b_indptr = mtx.indptr[c_begin:c_end+1] - b_begin
        nrow, ncol = mtx.shape[0], c_end-c_begin

        b_n_umi_col = n_umi_col[c_begin:c_end]
        pca_coord[c_begin:c_end,:] = fastrna_proj(
                b_data,
                b_indices,
                b_indptr,
                nrow,
                ncol,
                b_n_umi_col,
                eig_vec
                )

    pca_coord = pca_coord #/ np.sqrt(eig_val)[None,:]

    return eig_val[::-1], eig_vec[:,::-1], pca_coord[:,::-1], rrt
