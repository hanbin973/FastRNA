import numpy as np
cimport numpy as np

cdef extern from "mkl.h":

    # Sparse routines
    ctypedef int MKL_INT

    ctypedef enum sparse_index_base_t:
        SPARSE_INDEX_BASE_ZERO = 0
        SPARSE_INDEX_BASE_ONE = 1

    ctypedef enum sparse_status_t:
        SPARSE_STATUS_SUCCESS = 0 # the operation was successful
        SPARSE_STATUS_NOT_INITIALIZED = 1 # empty handle or matrix arrays
        SPARSE_STATUS_ALLOC_FAILED = 2 # internal error: memory allocation failed
        SPARSE_STATUS_INVALID_VALUE = 3 # invalid input value
        SPARSE_STATUS_EXECUTION_FAILED = 4 # e.g. 0-diagonal element for triangular solver, etc.
        SPARSE_STATUS_INTERNAL_ERROR = 5 # internal error
        SPARSE_STATUS_NOT_SUPPORTED = 6 # e.g. operation for double precision doesn't support other types */

    ctypedef enum sparse_operation_t:
        SPARSE_OPERATION_NON_TRANSPOSE = 10
        SPARSE_OPERATION_TRANSPOSE = 11
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12

    ctypedef enum sparse_matrix_type_t:
        SPARSE_MATRIX_TYPE_GENERAL = 20 # General case
        SPARSE_MATRIX_TYPE_SYMMETRIC = 21 # Triangular part of the matrix is to be processed
        SPARSE_MATRIX_TYPE_HERMITIAN = 22
        SPARSE_MATRIX_TYPE_TRIANGULAR = 23
        SPARSE_MATRIX_TYPE_DIAGONAL = 24 # diagonal matrix; only diagonal elements will be processed
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 25
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 26 # block-diagonal matrix; only diagonal blocks will be processed

    ctypedef enum sparse_fill_mode_t:
        SPARSE_FILL_MODE_LOWER = 40 # lower triangular part of the matrix is stored
        SPARSE_FILL_MODE_UPPER = 41 # upper triangular part of the matrix is stored
        SPARSE_FILL_MODE_FULL = 42 # upper triangular part of the matrix is stored

    ctypedef enum sparse_diag_type_t:
        SPARSE_DIAG_NON_UNIT = 50 # triangular matrix with non-unit diagonal
        SPARSE_DIAG_UNIT = 51 # triangular matrix with unit diagonal

    ctypedef enum sparse_layout_t:
        SPARSE_LAYOUT_ROW_MAJOR = 101 # C-style
        SPARSE_LAYOUT_COLUMN_MAJOR = 102 # Fortran-style

    struct sparse_matrix:
        pass

    ctypedef sparse_matrix* sparse_matrix_t

    struct matrix_descr:
        sparse_matrix_type_t type # matrix type: general, diagonal or triangular / symmetric / hermitian
        sparse_fill_mode_t mode # upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case)
        sparse_diag_type_t diag # unit or non-unit diagonal ( for triangular / symmetric / hermitian case)

    sparse_status_t mkl_sparse_s_create_csr(
        sparse_matrix_t* A,
        const sparse_index_base_t indexing, # indexing: C-style or Fortran-style
        const MKL_INT rows,
        const MKL_INT cols,
        MKL_INT *rows_start,
        MKL_INT *rows_end,
        MKL_INT *col_indx,
        float *values
    )

    sparse_status_t mkl_sparse_s_create_csc(
        sparse_matrix_t* A,
        const sparse_index_base_t indexing,
        const MKL_INT rows,
        const MKL_INT cols,
        MKL_INT *cols_start,
        MKL_INT *cols_end,
        MKL_INT *row_indx,
        float *values
    )

    sparse_status_t mkl_sparse_s_export_csr(
        const sparse_matrix_t source,
        sparse_index_base_t *indexing,
        MKL_INT *rows,
        MKL_INT *cols,
        MKL_INT **rows_start,
        MKL_INT **rows_end,
        MKL_INT **col_indx,
        float **values
    )

    sparse_status_t mkl_sparse_s_export_csc(
        const sparse_matrix_t source,
        sparse_index_base_t *indexing,
        MKL_INT *rows,
        MKL_INT *cols,
        MKL_INT **cols_start,
        MKL_INT **cols_end,
        MKL_INT **row_indx,
        float **values
    )

    sparse_status_t mkl_sparse_s_mv(
        const sparse_operation_t operation,
        const float alpha,
        const sparse_matrix_t A,
        const matrix_descr descr,
        const float *x,
        const float beta,
        float *y
    )

    sparse_status_t mkl_sparse_s_mm(
        const sparse_operation_t operation,
        const float alpha,
        const sparse_matrix_t A,
        const matrix_descr descr,
        const sparse_layout_t layout,
        const float* B,
        const MKL_INT cols,
        const MKL_INT ldb,
        const float beta,
        float* C,
        const MKL_INT ldc
    )

    sparse_status_t mkl_sparse_spmm(
        const sparse_operation_t operation,
        const sparse_matrix_t A,
        const sparse_matrix_t B,
        sparse_matrix_t *C
    )

    sparse_status_t mkl_sparse_syrk(
        const sparse_operation_t operation,
        const sparse_matrix_t A,
        sparse_matrix_t *C
    ) 

    sparse_status_t mkl_sparse_destroy(
            sparse_matrix_t A
    )

    sparse_status_t mkl_sparse_order(
            const sparse_matrix_t A
    )

    # Dense routines
    ctypedef enum CBLAS_LAYOUT:
        CblasRowMajor = 101
        CblasColMajor = 102
    
    void cblas_sger(
            const CBLAS_LAYOUT Layout,
            const MKL_INT m,
            const MKL_INT n,
			const float alpha,
            const float* x,
            const MKL_INT incx,
            const float* y,
            const MKL_INT incy,
            float* a,
            const MKL_INT lda
    )

    
cdef np.ndarray[np.float32_t, ndim=1] mkl_sparse_mv(
		const float[:],
		const int[:],
		const int[:],
		int,
		int,
		bint,
		const float[:],
		bint
		)

cpdef np.ndarray[np.float32_t, ndim=2] mkl_sparse_gram(
        const float[:],
        const int[:],
        const int[:],
        int,
        int,
        bint
        )

cdef np.ndarray[np.float32_t, ndim=2] cblas_ger(
        const float[:],
        const float[:]
        )
