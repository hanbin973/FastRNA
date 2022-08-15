cdef void spmatrix_to_dense_csc(
        const float[:],
        const int[:],
        const int[:],
        const int[:],
        int nrow,
        int ncol,
        float[::1, :]
        ) nogil

cdef void spmatrix_to_dense_csr(
        const float[:],
        const int[:],
        const int[:],
        const int[:],
        int nrow,
        int ncol,
        float[:, ::1]
        ) nogil

cdef float[:] csc_div_vec_row(
		const float[:],
		const int[:],
		const int[:],
		const float[:]
		)

cdef float[:] csc_div_vec_col(
		const float[:],
		const int[:],
		const int[:],
		const float[:]
		)

cdef float[:] mul(
		const float[:],
		const float[:]
		)

cdef float[:] div(
		const float[:],
		const float[:]
		)

cdef float[:] norm1(
		const float[:]
		)

cdef float[:] inv(
		const float[:]
		)

cdef float[:] pcomp(
		const float[:]
		)

cdef float[:] sqrt_c(
		const float[:]
		)

cdef float[:] mul_s(
        const float[:],
        const float
        )

cdef float vsum(
        const float[:]
        )
