import numpy as np
cimport numpy as np

import cython
from cython.parallel import prange

from fastrna.utils cimport *

# Conversion functions
cdef struct matrix_descr:
    sparse_matrix_type_t type

cdef sparse_matrix_t to_mkl_spmatrix(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        bint sptype
        ):

    cdef MKL_INT rows = nrow
    cdef MKL_INT cols = ncol
    cdef sparse_matrix_t A
    cdef sparse_index_base_t base_index = SPARSE_INDEX_BASE_ZERO

    cdef MKL_INT* start = &indptr[0]
    cdef MKL_INT* end = &indptr[1]
    cdef MKL_INT* index = &indices[0]
    cdef float* values = &data[0]

    if sptype:
        mkl_sparse_s_create = mkl_sparse_s_create_csr
    else:
        mkl_sparse_s_create = mkl_sparse_s_create_csc

    cdef sparse_status_t create_status = mkl_sparse_s_create(
            &A,
            base_index,
            rows,
            cols,
            start,
            end,
            index,
            values
            )

    return A

cdef np.ndarray[np.float32_t, ndim=2] to_python_dmatrix(
        sparse_matrix_t A,
        bint sptype
        ):

    cdef MKL_INT rows
    cdef MKL_INT cols
    cdef sparse_index_base_t base_index = SPARSE_INDEX_BASE_ZERO
    cdef MKL_INT* start
    cdef MKL_INT* end
    cdef MKL_INT* index
    cdef float* values
    cdef MKL_INT nptr

    if sptype:
        mkl_sparse_s_export = mkl_sparse_s_export_csr
        order = 'C'
    else:
        mkl_sparse_s_export = mkl_sparse_s_export_csc
        order = 'F'

    export_status = mkl_sparse_s_export(
            A,
            &base_index,
            &rows,
            &cols,
            &start,
            &end,
            &index,
            &values
            )

    if sptype:
        nptr = rows
    else:
        nptr = cols

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros(
            (rows, cols),
            dtype=np.float32,
            order=order
            )
    cdef int nnz = start[nptr]
    if sptype:
        spmatrix_to_dense_csr(
                <float[:nnz]> values,
                <int[:nnz]> index,
                <int[:nptr]> start,
                <int[:nptr]> end,
                rows,
                cols,
                result
                )
    else:
        spmatrix_to_dense_csc(
                <float[:nnz]> values,
                <int[:nnz]> index,
                <int[:nptr]> start,
                <int[:nptr]> end,
                rows,
                cols,
                result
                )

    return result


# Sparse routines
cdef np.ndarray[np.float32_t, ndim=1] mkl_sparse_mv(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        bint sptype,
        const float[:] vec,
        bint transpose
        ):

    cdef sparse_operation_t operation
    cdef MKL_INT shape_out
    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
        shape_out = ncol
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE
        shape_out = nrow

    cdef sparse_matrix_t A = to_mkl_spmatrix(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            sptype
            )

    cdef float alpha = 1.
    cdef float beta = 0.
    cdef matrix_descr mat_descript
    mat_descript.type = SPARSE_MATRIX_TYPE_GENERAL

    cdef np.ndarray[np.float32_t, ndim=1] result = np.zeros(shape_out, dtype=np.float32)
    cdef float[:] result_view = result

    status = mkl_sparse_s_mv(
            operation,
            alpha,
            A,
            mat_descript,
            &vec[0],
            beta,
            &result_view[0]
            )

    return result

cpdef np.ndarray[np.float32_t, ndim=2] mkl_sparse_gram(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        bint transpose
        ):

    cdef sparse_operation_t operation
    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE

    cdef sparse_matrix_t A = to_mkl_spmatrix(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            1
            )
    mkl_sparse_order(A)
    cdef sparse_matrix_t C

    status = mkl_sparse_syrk(
            operation,
            A,
            &C
            )

    cdef np.ndarray[np.float32_t, ndim=2] result = to_python_dmatrix(C, 1)

    mkl_sparse_destroy(A)
    mkl_sparse_destroy(C)

    return result

cdef np.ndarray[np.float32_t, ndim=2] mkl_sparse_mm(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        bint sptype,
        const float[:,:] mat,
        bint transpose
        ):

    cdef sparse_operation_t operation
    cdef MKL_INT shape_out
    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
        shape_out = ncol
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE
        shape_out = nrow

    cdef sparse_matrix_t A = to_mkl_spmatrix(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            sptype
            )

    cdef float alpha = 1.
    cdef float beta = 0.
    cdef matrix_descr mat_descript
    mat_descript.type = SPARSE_MATRIX_TYPE_GENERAL

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros(
            (shape_out, mat.shape[1]),
            dtype=np.float32
            )
    cdef float[:,:] result_view = result

    status = mkl_sparse_s_mm(
            operation,
            alpha,
            A,
            mat_descript,
            SPARSE_LAYOUT_ROW_MAJOR,
            &mat[0,0],
            mat.shape[1],
            mat.shape[1],
            beta,
            &result_view[0,0],
            mat.shape[1]
            )

    return result

# Dense routines
cdef np.ndarray[np.float32_t, ndim=2] cblas_ger(
        const float[:] x,
        const float[:] y
        ):

    cdef CBLAS_LAYOUT Layout = CblasRowMajor
    cdef MKL_INT m = x.shape[0]
    cdef MKL_INT n = y.shape[0]
    cdef float alpha = 1.
    cdef MKL_INT incx = 1
    cdef MKL_INT incy = 1
    cdef MKL_INT lda = n

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros(
            (m, n),
            dtype=np.float32
            )
    cdef float[:,:] result_view = result
    cblas_sger(
            Layout,
            m,
            n,
            alpha,
            &x[0],
            incx,
            &y[0],
            incy,
            &result_view[0,0],
            lda
            )

    return result
