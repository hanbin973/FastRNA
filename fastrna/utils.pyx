import cython
from cython.parallel import prange
from libc.math cimport sqrt

import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void spmatrix_to_dense_csc(
        const float[:] data,
        const int[:] indices,
        const int[:] ptrB,
        const int[:] ptrC,
        int nrow,
        int ncol,
        float[::1, :] result
        ) nogil:

    cdef Py_ssize_t i, i_end, i_begin, j
    for j in range(ncol):
        i_begin = ptrB[j]
        i_end = ptrC[j]
        for i in range(i_begin, i_end):
            result[indices[i], j] = data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void spmatrix_to_dense_csr(
        const float[:] data,
        const int[:] indices,
        const int[:] ptrB,
        const int[:] ptrC,
        int nrow,
        int ncol,
        float[:, ::1] result
        ) nogil:

    cdef Py_ssize_t i, i_end, i_begin, j
    for j in range(nrow):
        i_begin = ptrB[j]
        i_end = ptrC[j]
        for i in range(i_begin, i_end):
            result[j, indices[i]] = data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] csc_div_vec_row(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        const float[:] vec
        ):

    cdef float[:] result = np.empty(data.shape[0], dtype=np.float32)
    cdef Py_ssize_t nnz = data.shape[0]
    cdef Py_ssize_t i
    with nogil:
        for i in prange(nnz):
            result[i] = data[i] / vec[indices[i]]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] csc_div_vec_col(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        const float[:] vec
        ):

    cdef float[:] result = np.empty(data.shape[0], dtype=np.float32)
    cdef Py_ssize_t ncol = indptr.shape[0] - 1
    cdef Py_ssize_t i, i_begin, i_end, j
    with nogil:
        for j in prange(ncol):
            i_begin = indptr[j]
            i_end = indptr[j+1]
            for i in range(i_begin, i_end):
                result[i] = data[i] / vec[j]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] mul(
        const float[:] vec1,
        const float[:] vec2
        ):

    cdef Py_ssize_t i
    cdef float[:] result = np.empty(vec1.shape[0], dtype=np.float32)
    with nogil:
        for i in prange(vec1.shape[0]):
            result[i] = vec1[i] * vec2[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] div(
        const float[:] vec1,
        const float[:] vec2
        ):

    cdef Py_ssize_t i
    cdef float[:] result = np.empty(vec1.shape[0], dtype=np.float32)
    with nogil:
        for i in prange(vec1.shape[0]):
            result[i] = vec1[i] / vec2[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] norm1(
        const float[:] vec
        ):

    cdef Py_ssize_t i
    cdef float[:] result = np.empty(vec.shape[0], dtype=np.float32)
    cdef float num = 0
    with nogil:
        for i in prange(vec.shape[0]):
            num += vec[i]
        for i in prange(vec.shape[0]):
            result[i] = vec[i] / num

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] inv(
        const float[:] vec
        ):

    cdef Py_ssize_t i
    cdef float[:] result = np.empty(vec.shape[0], dtype=np.float32)
    with nogil:
        for i in prange(vec.shape[0]):
            result[i] = 1 / vec[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] pcomp(
        const float[:] vec
        ):   

    cdef Py_ssize_t i
    cdef float[:] result = np.empty(vec.shape[0], dtype=np.float32)
    with nogil:
        for i in prange(vec.shape[0]):
            result[i] = 1 - vec[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] sqrt_c(
        const float[:] vec
        ):   

    cdef Py_ssize_t i
    cdef float[:] result = np.empty(vec.shape[0], dtype=np.float32)
    with nogil:
        for i in prange(vec.shape[0]):
            result[i] = sqrt(vec[i])

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] mul_s(
        const float[:] vec,
        const float s
        ):   

    cdef Py_ssize_t i
    cdef float[:] result = np.empty(vec.shape[0], dtype=np.float32)
    with nogil:
        for i in prange(vec.shape[0]):
            result[i] = vec[i] * s

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float vsum(
        const float[:] vec
        ):   

    cdef Py_ssize_t i
    cdef float result = 0
    with nogil:
        for i in prange(vec.shape[0]):
            result += vec[i]

    return result

