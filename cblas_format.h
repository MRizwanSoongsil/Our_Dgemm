#pragma once

#include <stdint.h>

#ifdef USE_OPENBLAS_HEADER
#include "cblas.h"
#else
#include <mkl_cblas.h>
#endif

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
				CBLAS_TRANSPOSE TransB, const int64_t M, const int64_t N,
				const int64_t K, const double alpha, const double *A,
				const int64_t lda, const double *B, const int64_t ldb,
				const double beta, double *C, const int64_t ldc);
