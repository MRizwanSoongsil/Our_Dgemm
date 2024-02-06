#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdio.h>


#define MR 8
#define NR 20
#define MB 16
#define NB 40
#define KB 1000

#define NU 1

#define L1_DIST_A 192
#define L1_DIST_B 340

#define NT1 40


static void micro_kernel0(const int kk, const double *_A, const double *_B,
                          double *C, const int ldc) {
    __m512d _A0;
    __m512d _C0, _C1, _C2, _C3, _C4, _C5, _C6, _C7, _C8, _C9, _C10, _C11, _C12,
        _C13, _C14, _C15, _C16,_C17,_C18,_C19;
    _C0 = _mm512_loadu_pd(&C[0 * ldc]);
    _C1 = _mm512_loadu_pd(&C[1 * ldc]);
    _C2 = _mm512_loadu_pd(&C[2 * ldc]);
    _C3 = _mm512_loadu_pd(&C[3 * ldc]);
    _C4 = _mm512_loadu_pd(&C[4 * ldc]);
    _C5 = _mm512_loadu_pd(&C[5 * ldc]);
    _C6 = _mm512_loadu_pd(&C[6 * ldc]);
    _C7 = _mm512_loadu_pd(&C[7 * ldc]);
    _C8 = _mm512_loadu_pd(&C[8 * ldc]);
    _C9 = _mm512_loadu_pd(&C[9 * ldc]);
    _C10 = _mm512_loadu_pd(&C[10 * ldc]);
    _C11 = _mm512_loadu_pd(&C[11 * ldc]);
    _C12 = _mm512_loadu_pd(&C[12 * ldc]);
    _C13 = _mm512_loadu_pd(&C[13 * ldc]);
    _C14 = _mm512_loadu_pd(&C[14 * ldc]);
    _C15 = _mm512_loadu_pd(&C[15 * ldc]);
   _C16 = _mm512_loadu_pd(&C[16 * ldc]);
    _C17 = _mm512_loadu_pd(&C[17 * ldc]);
    _C18 = _mm512_loadu_pd(&C[18 * ldc]);
    _C19 = _mm512_loadu_pd(&C[19 * ldc]);

#pragma unroll(NU)
    for (int i = 0; i < kk; ++i) {
        _mm_prefetch((const void *)&_A[L1_DIST_A + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 8], _MM_HINT_T0);
        _A0 = _mm512_loadu_pd(&_A[0]);
        _C0 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[0]), _A0, _C0);
        _C1 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[1]), _A0, _C1);
        _C2 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[2]), _A0, _C2);
        _C3 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[3]), _A0, _C3);
        _C4 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[4]), _A0, _C4);
        _C5 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[5]), _A0, _C5);
        _C6 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[6]), _A0, _C6);
        _C7 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[7]), _A0, _C7);
        _C8 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[8]), _A0, _C8);
        _C9 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[9]), _A0, _C9);
        _C10 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[10]), _A0, _C10);
        _C11 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[11]), _A0, _C11);
        _C12 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[12]), _A0, _C12);
        _C13 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[13]), _A0, _C13);
        _C14 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[14]), _A0, _C14);
        _C15 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[15]), _A0, _C15);
	_C16 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[16]), _A0, _C16);
   	_C17 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[17]), _A0, _C17);
   	_C18 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[18]), _A0, _C18);
    	_C19 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[19]), _A0, _C19);
        _A += MR;
        _B += NR;
    }
    _mm512_storeu_pd(&C[0 * ldc], _C0);
    _mm512_storeu_pd(&C[1 * ldc], _C1);
    _mm512_storeu_pd(&C[2 * ldc], _C2);
    _mm512_storeu_pd(&C[3 * ldc], _C3);
    _mm512_storeu_pd(&C[4 * ldc], _C4);
    _mm512_storeu_pd(&C[5 * ldc], _C5);
    _mm512_storeu_pd(&C[6 * ldc], _C6);
    _mm512_storeu_pd(&C[7 * ldc], _C7);
    _mm512_storeu_pd(&C[8 * ldc], _C8);
    _mm512_storeu_pd(&C[9 * ldc], _C9);
    _mm512_storeu_pd(&C[10 * ldc], _C10);
    _mm512_storeu_pd(&C[11 * ldc], _C11);
    _mm512_storeu_pd(&C[12 * ldc], _C12);
    _mm512_storeu_pd(&C[13 * ldc], _C13);
    _mm512_storeu_pd(&C[14 * ldc], _C14);
    _mm512_storeu_pd(&C[15 * ldc], _C15);
  _mm512_storeu_pd(&C[16 * ldc], _C16);
  _mm512_storeu_pd(&C[17 * ldc], _C17);
  _mm512_storeu_pd(&C[18 * ldc], _C18);
  _mm512_storeu_pd(&C[19 * ldc], _C19);

}

static void micro_kernel1(const int kk, const double *_A, const double *_B,
                          double *_C) {
    __m512d _A0;
    __m512d _C0, _C1, _C2, _C3, _C4, _C5, _C6, _C7, _C8, _C9, _C10, _C11, _C12,
        _C13, _C14, _C15, _C16,_C17,_C18,_C19;
    _C0 = _mm512_setzero_pd();
    _C1 = _mm512_setzero_pd();
    _C2 = _mm512_setzero_pd();
    _C3 = _mm512_setzero_pd();
    _C4 = _mm512_setzero_pd();
    _C5 = _mm512_setzero_pd();
    _C6 = _mm512_setzero_pd();
    _C7 = _mm512_setzero_pd();
    _C8 = _mm512_setzero_pd();
    _C9 = _mm512_setzero_pd();
    _C10 = _mm512_setzero_pd();
    _C11 = _mm512_setzero_pd();
    _C12 = _mm512_setzero_pd();
    _C13 = _mm512_setzero_pd();
    _C14 = _mm512_setzero_pd();
    _C15 = _mm512_setzero_pd();
   _C16 = _mm512_setzero_pd();
    _C17 = _mm512_setzero_pd();
    _C18 = _mm512_setzero_pd();
    _C19 = _mm512_setzero_pd();

#pragma unroll(NU)
    for (int i = 0; i < kk; ++i) {
        _mm_prefetch((const void *)&_A[L1_DIST_A + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 8], _MM_HINT_T0);
        _A0 = _mm512_loadu_pd(&_A[0]);
        _C0 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[0]), _A0, _C0);
        _C1 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[1]), _A0, _C1);
        _C2 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[2]), _A0, _C2);
        _C3 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[3]), _A0, _C3);
        _C4 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[4]), _A0, _C4);
        _C5 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[5]), _A0, _C5);
        _C6 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[6]), _A0, _C6);
        _C7 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[7]), _A0, _C7);
        _C8 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[8]), _A0, _C8);
        _C9 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[9]), _A0, _C9);
        _C10 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[10]), _A0, _C10);
        _C11 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[11]), _A0, _C11);
        _C12 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[12]), _A0, _C12);
        _C13 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[13]), _A0, _C13);
        _C14 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[14]), _A0, _C14);
	_C16 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[16]), _A0, _C16);
    _C17 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[17]), _A0, _C17);
    _C18 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[18]), _A0, _C18);
    _C19 = _mm512_fnmadd_pd(_mm512_set1_pd(_B[19]), _A0, _C19);

        _A += MR;
        _B += NR;
    }
    _mm512_storeu_pd(&_C[0 * MR], _C0);
    _mm512_storeu_pd(&_C[1 * MR], _C1);
    _mm512_storeu_pd(&_C[2 * MR], _C2);
    _mm512_storeu_pd(&_C[3 * MR], _C3);
    _mm512_storeu_pd(&_C[4 * MR], _C4);
    _mm512_storeu_pd(&_C[5 * MR], _C5);
    _mm512_storeu_pd(&_C[6 * MR], _C6);
    _mm512_storeu_pd(&_C[7 * MR], _C7);
    _mm512_storeu_pd(&_C[8 * MR], _C8);
    _mm512_storeu_pd(&_C[9 * MR], _C9);
    _mm512_storeu_pd(&_C[10 * MR], _C10);
    _mm512_storeu_pd(&_C[11 * MR], _C11);
    _mm512_storeu_pd(&_C[12 * MR], _C12);
    _mm512_storeu_pd(&_C[13 * MR], _C13);
    _mm512_storeu_pd(&_C[14 * MR], _C14);
    _mm512_storeu_pd(&_C[15 * MR], _C15);
   _mm512_storeu_pd(&_C[16 * MR], _C16);
  _mm512_storeu_pd(&_C[17 * MR], _C17);
  _mm512_storeu_pd(&_C[18 * MR], _C18);
  _mm512_storeu_pd(&_C[19 * MR], _C19);

}

static void micro_dxpy(int mmm, int nnn, double *C, const double *_C, int ldc) {
    int i;
    for (i = 0; i < nnn; ++i) {
        C [0:mmm] += _C [i * MR:mmm];
        C += ldc;
    }
}

static void packarc(const int mm, const int kk, const double *A_, const int lda,
                    double *_A) {
    const int q = mm / MR;
    const int r = mm % MR;

    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < kk; ++j) {
            _A [j * MR + i * kk * MR:MR] = A_ [j + i * MR * lda:MR:lda];
        }
    }
    _A += q * kk * MR;
    A_ += q * lda * MR;
    if (r > 0) {
        for (int j = 0; j < kk; ++j) {
            _A [0:r] = A_ [0:r:lda];
            _A [r:MR - r] = 0.0;
            _A += MR;
            A_ += 1;
        }
    }
}


static void packacc(int           mm, // # of rows
		int           kk, // # of cols
		const double *A_, // from *A_
		int           lda, // distance
		double       *_A) // to *_A
{
	int q = mm / MR; // quotient
	int r = mm % MR; // remainder
	int i, j;
	
	for(j = 0; j < q; ++j)
	{
		for(i = 0; i < kk; ++i)
		{
			_A[i*MR+j*kk*MR:MR] = A_[i*lda+j*MR:MR];
		}
	}
	_A += q*kk*MR;
	A_ += q*MR;
	if(r > 0)
	{
		for(i = 0; i < kk; ++i)
		{
			_A[0:r] = A_[0:r];
			_A[r:MR-r] = 0.0;
			_A += MR;
			A_ += lda;
		}
	}
}

static void packbcr(const int kk, const int nn, const double *B_, const int ldb,
                    double *_B) {
    int q = nn / NR;
    int r = nn % NR;
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < kk; ++j) {
            _B [j * NR + i * kk * NR:NR] = B_ [j + i * NR * ldb:NR:ldb];
        }
    }
    _B += q * kk * NR;
    B_ += q * ldb * NR;
    if (r > 0) {
        for (int j = 0; j < kk; ++j) {
            _B [0:r] = B_ [0:r:ldb];
            _B [r:NR - r] = 0.0;
            _B += NR;
            B_ += 1;
        }
    }
}


void userdgemm_(const char *transa, const char *transb, const int *_m,
                   const int *_n, const int *_k, const double *_alpha,
                   const double *restrict A, const int *_lda,
                   const double *restrict B, const int *_ldb,
                   const double *_beta, double *restrict C, const int *_ldc) {
    const int m = *_m;
    const int n = *_n;
    const int k = *_k;
    const double alpha = *_alpha;
    const int lda = *_lda;
    const int ldb = *_ldb;
    const double beta = *_beta;
    const int ldc = *_ldc;

	const int mc = (m + MB - 1) / MB;
	const int mr = m % MB;
	const int kc = (k + KB - 1) / KB;
	const int kr = k % KB;
	const int nc = (n + NB - 1) / NB;
	const int nr = n % NB;
	
    double *_B;
   _B = (double*)malloc(sizeof(double)  * KB * NB);

    for (int ni = 0; ni < nc; ++ni) {
        const int nn = (ni != nc - 1 || nr == 0) ? NB : nr;
        const int nnc = (nn + NR - 1) / NR;
        const int nnr = nn % NR;
	for (int ki = 0; ki < kc; ++ki) {
            const int kk = (ki != kc - 1 || kr == 0) ? KB : kr;
	    packbcr(kk, nn, &B[ki * KB + ni * NB * ldb], ldb, _B);
#pragma omp parallel num_threads(NT1)
        {
         double* _A;
	 double* _C;
				
	_A = (double*)malloc(sizeof(double) * MB * KB);
	_C = (double*)malloc(sizeof(double) * MR * NR);
#pragma omp for schedule(runtime)
            for (int mi = 0; mi < mc; ++mi) {
                const int mm = (mi != mc - 1 || mr == 0) ? MB : mr;
                const int mmc = (mm + MR - 1) / MR;
                const int mmr = mm % MR;
                	 
                    packarc(mm, kk, &A[mi * MB * lda + ki * KB], lda, _A);
                    //packacc(mm, kk, &A[mi * MB + ki * KB * lda], lda, _A);

                    for (int nni = 0; nni < nnc; ++nni) {
                        const int nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;
                        for (int mmi = 0; mmi < mmc; ++mmi) {
                            const int mmm =
                                (mmi != mmc - 1 || mmr == 0) ? MR : mmr;
                            if (mmm == MR && nnn == NR) {
                                micro_kernel0(
                                    kk, &_A[mmi * MR * kk], &_B[nni * NR * kk],
                                    &C[ni * NB * ldc + nni * NR * ldc +
                                       mi * MB + mmi * MR],
                                    ldc);
                            } else {
                                micro_kernel1(kk, &_A[mmi * MR * kk],
                                              &_B[nni * NR * kk], _C);
                                micro_dxpy(mmm, nnn,
                                           &C[ni * NB * ldc + nni * NR * ldc +
                                              mi * MB + mmi * MR],
                                           _C, ldc);
                            }
                        }
                    }
                }
		free(_A);        	
        	free(_C);
            }
            
        }
	
    }
    free(_B);
}
