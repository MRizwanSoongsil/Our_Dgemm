#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdio.h>
#include "x86intrin.h"
#include "zmmintrin.h"


#define MR 16
#define NR 14
#define MB 1024
#define NB 84
#define KB 500


#define L1_DIST_A 320
#define L1_DIST_B 280

#ifndef NT
#define NT 40
#endif
#ifndef NT1
#define NT1 1
#endif

void micro_kernel0(int k,
  const double * A,
    const double * B, double * C, int ncol) {
  int i;
  register __m512d _A0, _A1;
  register __m512d _C0_0, _C1_0, _C2_0, _C3_0, _C4_0, _C5_0, _C6_0, _C7_0, _C8_0, _C9_0, _C10_0, _C11_0, _C12_0, _C13_0;
  register __m512d _C0_1, _C1_1, _C2_1, _C3_1, _C4_1, _C5_1, _C6_1, _C7_1, _C8_1, _C9_1, _C10_1, _C11_1, _C12_1, _C13_1;
  _C0_0 = _mm512_loadu_pd( & C[0 * ncol + 0]);
  _C0_1 = _mm512_loadu_pd( & C[0 * ncol + 8]);
  _C1_0 = _mm512_loadu_pd( & C[1 * ncol + 0]);
  _C1_1 = _mm512_loadu_pd( & C[1 * ncol + 8]);
  _C2_0 = _mm512_loadu_pd( & C[2 * ncol + 0]);
  _C2_1 = _mm512_loadu_pd( & C[2 * ncol + 8]);
  _C3_0 = _mm512_loadu_pd( & C[3 * ncol + 0]);
  _C3_1 = _mm512_loadu_pd( & C[3 * ncol + 8]);
  _C4_0 = _mm512_loadu_pd( & C[4 * ncol + 0]);
  _C4_1 = _mm512_loadu_pd( & C[4 * ncol + 8]);
  _C5_0 = _mm512_loadu_pd( & C[5 * ncol + 0]);
  _C5_1 = _mm512_loadu_pd( & C[5 * ncol + 8]);
  _C6_0 = _mm512_loadu_pd( & C[6 * ncol + 0]);
  _C6_1 = _mm512_loadu_pd( & C[6 * ncol + 8]);
  _C7_0 = _mm512_loadu_pd( & C[7 * ncol + 0]);
  _C7_1 = _mm512_loadu_pd( & C[7 * ncol + 8]);
  _C8_0 = _mm512_loadu_pd( & C[8 * ncol + 0]);
  _C8_1 = _mm512_loadu_pd( & C[8 * ncol + 8]);
  _C9_0 = _mm512_loadu_pd( & C[9 * ncol + 0]);
  _C9_1 = _mm512_loadu_pd( & C[9 * ncol + 8]);
  _C10_0 = _mm512_loadu_pd( & C[10 * ncol + 0]);
  _C10_1 = _mm512_loadu_pd( & C[10 * ncol + 8]);
  _C11_0 = _mm512_loadu_pd( & C[11 * ncol + 0]);
  _C11_1 = _mm512_loadu_pd( & C[11 * ncol + 8]);
  _C12_0 = _mm512_loadu_pd( & C[12 * ncol + 0]);
  _C12_1 = _mm512_loadu_pd( & C[12 * ncol + 8]);
  _C13_0 = _mm512_loadu_pd( & C[13 * ncol + 0]);
  _C13_1 = _mm512_loadu_pd( & C[13 * ncol + 8]);
  // _C += A*B
  #pragma unroll(1)
  for (i = 0; i < k; i++) {
    // A L1 prefetch
    _mm_prefetch((const void * ) & A[L1_DIST_A + 0], _MM_HINT_T0);
    _mm_prefetch((const void * ) & A[L1_DIST_A + 8], _MM_HINT_T0);
    // B L1 prefetch
    _mm_prefetch((const void * ) & B[L1_DIST_B + 0], _MM_HINT_T0);
    _mm_prefetch((const void * ) & B[L1_DIST_B + 8], _MM_HINT_T0);
    _A0 = _mm512_loadu_pd( & A[0]);
    _A1 = _mm512_loadu_pd( & A[8]);
    _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A0, _C0_0);
    _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A1, _C0_1);
    _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A0, _C1_0);
    _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A1, _C1_1);
    _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A0, _C2_0);
    _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A1, _C2_1);
    _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A0, _C3_0);
    _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A1, _C3_1);
    _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A0, _C4_0);
    _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A1, _C4_1);
    _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A0, _C5_0);
    _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A1, _C5_1);
    _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A0, _C6_0);
    _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A1, _C6_1);
    _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A0, _C7_0);
    _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A1, _C7_1);
    _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A0, _C8_0);
    _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A1, _C8_1);
    _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A0, _C9_0);
    _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A1, _C9_1);
    _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A0, _C10_0);
    _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A1, _C10_1);
    _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A0, _C11_0);
    _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A1, _C11_1);
    _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A0, _C12_0);
    _C12_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A1, _C12_1);
    _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A0, _C13_0);
    _C13_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A1, _C13_1);
    A += MR;
    B += NR;
  }
  // store _C -> C
  _mm512_storeu_pd( & C[0 * ncol + 0], _C0_0);
  _mm512_storeu_pd( & C[0 * ncol + 8], _C0_1);
  _mm512_storeu_pd( & C[1 * ncol + 0], _C1_0);
  _mm512_storeu_pd( & C[1 * ncol + 8], _C1_1);
  _mm512_storeu_pd( & C[2 * ncol + 0], _C2_0);
  _mm512_storeu_pd( & C[2 * ncol + 8], _C2_1);
  _mm512_storeu_pd( & C[3 * ncol + 0], _C3_0);
  _mm512_storeu_pd( & C[3 * ncol + 8], _C3_1);
  _mm512_storeu_pd( & C[4 * ncol + 0], _C4_0);
  _mm512_storeu_pd( & C[4 * ncol + 8], _C4_1);
  _mm512_storeu_pd( & C[5 * ncol + 0], _C5_0);
  _mm512_storeu_pd( & C[5 * ncol + 8], _C5_1);
  _mm512_storeu_pd( & C[6 * ncol + 0], _C6_0);
  _mm512_storeu_pd( & C[6 * ncol + 8], _C6_1);
  _mm512_storeu_pd( & C[7 * ncol + 0], _C7_0);
  _mm512_storeu_pd( & C[7 * ncol + 8], _C7_1);
  _mm512_storeu_pd( & C[8 * ncol + 0], _C8_0);
  _mm512_storeu_pd( & C[8 * ncol + 8], _C8_1);
  _mm512_storeu_pd( & C[9 * ncol + 0], _C9_0);
  _mm512_storeu_pd( & C[9 * ncol + 8], _C9_1);
  _mm512_storeu_pd( & C[10 * ncol + 0], _C10_0);
  _mm512_storeu_pd( & C[10 * ncol + 8], _C10_1);
  _mm512_storeu_pd( & C[11 * ncol + 0], _C11_0);
  _mm512_storeu_pd( & C[11 * ncol + 8], _C11_1);
  _mm512_storeu_pd( & C[12 * ncol + 0], _C12_0);
  _mm512_storeu_pd( & C[12 * ncol + 8], _C12_1);
  _mm512_storeu_pd( & C[13 * ncol + 0], _C13_0);
  _mm512_storeu_pd( & C[13 * ncol + 8], _C13_1);
}
void micro_kernel1(int k,
  const double * A,
    const double * B, double * C, int ncol) {
  int i;
  register __m512d _A0, _A1;
  register __m512d _C0_0, _C1_0, _C2_0, _C3_0, _C4_0, _C5_0, _C6_0, _C7_0, _C8_0, _C9_0, _C10_0, _C11_0, _C12_0, _C13_0;
  register __m512d _C0_1, _C1_1, _C2_1, _C3_1, _C4_1, _C5_1, _C6_1, _C7_1, _C8_1, _C9_1, _C10_1, _C11_1, _C12_1, _C13_1;
  _C0_0 = _mm512_setzero_pd();
  _C0_1 = _mm512_setzero_pd();
  _C1_0 = _mm512_setzero_pd();
  _C1_1 = _mm512_setzero_pd();
  _C2_0 = _mm512_setzero_pd();
  _C2_1 = _mm512_setzero_pd();
  _C3_0 = _mm512_setzero_pd();
  _C3_1 = _mm512_setzero_pd();
  _C4_0 = _mm512_setzero_pd();
  _C4_1 = _mm512_setzero_pd();
  _C5_0 = _mm512_setzero_pd();
  _C5_1 = _mm512_setzero_pd();
  _C6_0 = _mm512_setzero_pd();
  _C6_1 = _mm512_setzero_pd();
  _C7_0 = _mm512_setzero_pd();
  _C7_1 = _mm512_setzero_pd();
  _C8_0 = _mm512_setzero_pd();
  _C8_1 = _mm512_setzero_pd();
  _C9_0 = _mm512_setzero_pd();
  _C9_1 = _mm512_setzero_pd();
  _C10_0 = _mm512_setzero_pd();
  _C10_1 = _mm512_setzero_pd();
  _C11_0 = _mm512_setzero_pd();
  _C11_1 = _mm512_setzero_pd();
  _C12_0 = _mm512_setzero_pd();
  _C12_1 = _mm512_setzero_pd();
  _C13_0 = _mm512_setzero_pd();
  _C13_1 = _mm512_setzero_pd();
  // _C += A*B
  #pragma unroll(1)
  for (i = 0; i < k; i++) {
    // A L1 prefetch
    _mm_prefetch((const void * ) & A[L1_DIST_A + 0], _MM_HINT_T0);
    _mm_prefetch((const void * ) & A[L1_DIST_A + 8], _MM_HINT_T0);
    // B L1 prefetch
    _mm_prefetch((const void * ) & B[L1_DIST_B + 0], _MM_HINT_T0);
    _mm_prefetch((const void * ) & B[L1_DIST_B + 8], _MM_HINT_T0);
    _A0 = _mm512_loadu_pd( & A[0]);
    _A1 = _mm512_loadu_pd( & A[8]);
    _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A0, _C0_0);
    _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A1, _C0_1);
    _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A0, _C1_0);
    _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A1, _C1_1);
    _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A0, _C2_0);
    _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A1, _C2_1);
    _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A0, _C3_0);
    _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A1, _C3_1);
    _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A0, _C4_0);
    _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A1, _C4_1);
    _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A0, _C5_0);
    _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A1, _C5_1);
    _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A0, _C6_0);
    _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A1, _C6_1);
    _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A0, _C7_0);
    _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A1, _C7_1);
    _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A0, _C8_0);
    _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A1, _C8_1);
    _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A0, _C9_0);
    _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A1, _C9_1);
    _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A0, _C10_0);
    _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A1, _C10_1);
    _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A0, _C11_0);
    _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A1, _C11_1);
    _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A0, _C12_0);
    _C12_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A1, _C12_1);
    _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A0, _C13_0);
    _C13_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A1, _C13_1);
    A += MR;
    B += NR;
  }
  // store _C -> C
  _mm512_storeu_pd( & C[0 * ncol + 0], _C0_0);
  _mm512_storeu_pd( & C[0 * ncol + 8], _C0_1);
  _mm512_storeu_pd( & C[1 * ncol + 0], _C1_0);
  _mm512_storeu_pd( & C[1 * ncol + 8], _C1_1);
  _mm512_storeu_pd( & C[2 * ncol + 0], _C2_0);
  _mm512_storeu_pd( & C[2 * ncol + 8], _C2_1);
  _mm512_storeu_pd( & C[3 * ncol + 0], _C3_0);
  _mm512_storeu_pd( & C[3 * ncol + 8], _C3_1);
  _mm512_storeu_pd( & C[4 * ncol + 0], _C4_0);
  _mm512_storeu_pd( & C[4 * ncol + 8], _C4_1);
  _mm512_storeu_pd( & C[5 * ncol + 0], _C5_0);
  _mm512_storeu_pd( & C[5 * ncol + 8], _C5_1);
  _mm512_storeu_pd( & C[6 * ncol + 0], _C6_0);
  _mm512_storeu_pd( & C[6 * ncol + 8], _C6_1);
  _mm512_storeu_pd( & C[7 * ncol + 0], _C7_0);
  _mm512_storeu_pd( & C[7 * ncol + 8], _C7_1);
  _mm512_storeu_pd( & C[8 * ncol + 0], _C8_0);
  _mm512_storeu_pd( & C[8 * ncol + 8], _C8_1);
  _mm512_storeu_pd( & C[9 * ncol + 0], _C9_0);
  _mm512_storeu_pd( & C[9 * ncol + 8], _C9_1);
  _mm512_storeu_pd( & C[10 * ncol + 0], _C10_0);
  _mm512_storeu_pd( & C[10 * ncol + 8], _C10_1);
  _mm512_storeu_pd( & C[11 * ncol + 0], _C11_0);
  _mm512_storeu_pd( & C[11 * ncol + 8], _C11_1);
  _mm512_storeu_pd( & C[12 * ncol + 0], _C12_0);
  _mm512_storeu_pd( & C[12 * ncol + 8], _C12_1);
  _mm512_storeu_pd( & C[13 * ncol + 0], _C13_0);
  _mm512_storeu_pd( & C[13 * ncol + 8], _C13_1);
}

// notation
// m = n*q + d
// quotient : Xq
// remainder: Xd

// Packing A, c-major -> c-major
void packacc(int row, // # of rows
  int col, // # of cols
  const double * mt, // from *mt
    int inc, // distance
    double * bk) // to *bk
{
  int q = row / MR; // quotient
  int r = row % MR; // remainder
  int i, j;

  for (j = 0; j < q; ++j) {
    for (i = 0; i < col; ++i) {
      // copy NR elements from &mt[i*inc+j*NR] with interval 1
      // loop order is j-loop -> i-loop to access bk contiguously
      bk[i * MR + j * col * MR: MR] = mt[i * inc + j * MR: MR];
    }
  }
  // move pointers
  bk += q * col * MR;
  mt += q * MR;
  // remainder part
  if (r > 0) {
    for (i = 0; i < col; ++i) {
      bk[0: r] = mt[0: r];
      bk[r: MR - r] = 0.0;
      bk += MR;
      mt += inc;
    }
  }
}
// Packing A, r-major -> c-major
void packarc(int row, // # of rows
  int col, // # of cols
  const double * mt, //  from *mt
    int inc, // distance
    double * bk) // to *bk
{
  int q = row / MR; // quotient
  int r = row % MR; // remainder
  int i, j;

  for (i = 0; i < q; ++i) {
    for (j = 0; j < col; ++j) {
      // copy MR elements from &mt[j+i*MR*inc] with interval inc
      // loop order is i-loop -> j-loop to access bk contiguously
      bk[j * MR + i * col * MR: MR] = mt[j + i * MR * inc: MR: inc];
    }
  }
  // move pointers
  bk += q * col * MR;
  mt += q * inc * MR;
  // remainder part
  if (r > 0) {
    for (j = 0; j < col; ++j) {
      bk[0: r] = mt[0: r: inc];
      bk[r: MR - r] = 0.0;
      bk += MR;
      mt += 1;
    }
  }
}

// Packing B, c-major -> r-major
void packbcr(int row, // # of rows
  int col, // # of cols
  const double * mt, //  from *mt
    int inc, // distance
    double * bk) // to *bk
{
  int q = col / NR; // quotient
  int r = col % NR; // remainder
  int i, j;

  for (i = 0; i < q; ++i) {
    for (j = 0; j < row; ++j) {
      // copy MR elements from &mt[j+i*MR*inc] with interval inc
      // loop order is i-loop -> j-loop to access bk contiguously
      bk[j * NR + i * row * NR: NR] = mt[j + i * NR * inc: NR: inc];
    }
  }
  // move pointers
  bk += q * row * NR;
  mt += q * inc * NR;
  // remainder part
  if (r > 0) {
    for (j = 0; j < row; ++j) {
      bk[0: r] = mt[0: r: inc];
      bk[r: NR - r] = 0.0;
      bk += NR;
      mt += 1;
    }
  }
}
// Packing B, r-major -> r-major
void packbrr(int row, // # of rows
  int col, // # of cols
  const double * mt, // from *mt
    int inc, // distance
    double * bk) // to *bk
{
  int q = col / NR; // quotient
  int r = col % NR; // remainder
  int i, j;

  for (j = 0; j < q; ++j) {
    for (i = 0; i < row; ++i) {
      // copy NR elements from &mt[i*inc+j*NR] with interval 1
      // loop order is j-loop -> i-loop to access bk contiguously
      bk[i * NR + j * row * NR: NR] = mt[i * inc + j * NR: NR];
    }
  }
  // move pointers
  bk += q * row * NR;
  mt += q * NR;
  // remainder part
  if (r > 0) {
    for (i = 0; i < row; ++i) {
      bk[0: r] = mt[0: r];
      bk[r: NR - r] = 0.0;
      bk += NR;
      mt += inc;
    }
  }
}
// Micro C = _C + C
void micro_dxpy(int m,
  int n,
  double * C,
  const double * D,
    int ncol) {
  int i;
  for (i = 0; i < n; ++i) {
    C[0: m] += D[i * MR: m];
    C += ncol;
  }
}



void zerodgemm_(const char *transa, const char *transb, const int *_m,
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

    int ki;

    double* _A;
    double* _B;
    double* _C;
     _A = (double*)malloc(sizeof(double) * MB * KB);
    for (ki = 0; ki < kc; ++ki) {
        const int kk = (ki != kc - 1 || kr == 0) ? KB : kr;
        int mi, ni, mmi, nni;

            for (mi = 0; mi < mc; ++mi) {
                const int mm = (mi != mc - 1 || mr == 0) ? MB : mr;
                const int mmc = (mm + MR - 1) / MR;
                const int mmr = mm % MR;
                packacc(mm, kk, &A[mi * MB + ki * KB * lda], lda, _A);
#pragma omp parallel num_threads(40) private( _B, _C, ni, mmi, nni)
{
            _B = (double*)malloc(sizeof(double) * KB * NB);
            _C = (double*)malloc(sizeof(double) * MR * NR);

#pragma omp for schedule(runtime)
                for (ni = 0; ni < nc; ++ni) {
                    const int nn = (ni != nc - 1 || nr == 0) ? NB : nr;
                    const int nnc = (nn + NR - 1) / NR;
                    const int nnr = nn % NR;
                    packbrr(kk, nn, &B[ki * KB * ldb + ni * NB], ldb, _B);

                    for (nni = 0; nni < nnc; ++nni) {
                        const int nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;

                        for (mmi = 0; mmi < mmc; ++mmi) {
                            const int mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;
                            if (mmm == MR && nnn == NR) {
                                micro_kernel0(
                                    kk, &_A[mmi * MR * kk], &_B[nni * NR * kk],
                                    &C[ni * NB * ldc + nni * NR * ldc + mi * MB + mmi * MR],
                                    ldc);
                            } else {
                                micro_kernel1(kk, &_A[mmi * MR * kk], &_B[nni * NR * kk], _C, MR);
                                micro_dxpy(mmm, nnn, &C[ni * NB * ldc + nni * NR * ldc + mi * MB + mmi * MR], _C, ldc);
                            }
                        }
                    }
                }
            
            free(_C);
            free(_B);
	   }
        }
    }
    free(_A);
}
