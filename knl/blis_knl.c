#ifndef FORTRAN_FORMAT
#include "cblas_format.h"
#endif

#include <mkl.h>
#define BLIS_DISABLE_BLAS_DEFS
#include "knl/blis.h"

#ifdef FORTRAN_FORMAT
void userdgemm_(const char *transa, const char *transb, const int *_m,
                const int *_n, const int *_k, const double *_alpha,
                const double *restrict A, const int *_lda,
                const double *restrict B, const int *_ldb,
                const double *_beta, double *restrict C, const int *_ldc) {
	const CBLAS_LAYOUT layout = CblasColMajor;
	const CBLAS_TRANSPOSE TransA = (transa[0] == 'T') ? CblasTrans : CblasNoTrans;
	const CBLAS_TRANSPOSE TransB = (transb[0] == 'T') ? CblasTrans : CblasNoTrans;
    const int m = *_m;
    const int n = *_n;
    const int k = *_k;
    const double alpha = *_alpha;
    const int lda = *_lda;
    const int ldb = *_ldb;
    const double beta = *_beta;
    const int ldc = *_ldc;
#else
void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
				CBLAS_TRANSPOSE TransB, const int64_t m, const int64_t n,
				const int64_t k, const double alpha, const double *A,
				const int64_t lda, const double *B, const int64_t ldb,
				const double beta, double *C, const int64_t ldc) {
#endif
    inc_t rsa, csa;
    inc_t rsb, csb;
    inc_t rsc, csc;

    rsc = 1;
    csc = m;
    rsa = 1;
    csa = (TransA == CblasTrans? k : m);
    rsb = 1;
    csb = (TransB == CblasTrans? n : k);

    bli_dgemm(TransA == CblasTrans? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              TransB == CblasTrans? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              m, n, k, &alpha, A, rsa, csa, B, rsb, csb, &beta, C, rsc, csc);
}
