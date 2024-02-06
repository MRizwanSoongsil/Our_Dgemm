#ifdef USE_HBWMALLOC
#include <hbwmalloc.h>
#define ALLOCATE(ptr, size) hbw_posix_memalign((void**)&(ptr), 64, (size))
#define FREE(ptr) hbw_free(ptr)
#else
#define ALLOCATE(ptr, size) ptr = malloc((size))
#define FREE(ptr) free(ptr)
#endif

#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "cblas_format.h"

#ifndef KERNEL
#define KERNEL "UNKNOWN"
#endif

static void set_data(double *matrix, int size, int seed, double min_value, double max_value);

int main(int argc, char **argv) {
    if (argc != 10) {
        fprintf(stderr, "Usage : %s Layout(Row/Col) TransA(T/N) TransB(T/N) M N K alpha beta iteration\n", argv[0]);
        return 1;
    }

    const CBLAS_LAYOUT layout = (argv[1][0] == 'R') ? CblasRowMajor : CblasColMajor;
    const CBLAS_TRANSPOSE TransA = (argv[2][0] == 'T') ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE TransB = (argv[3][0] == 'T') ? CblasTrans : CblasNoTrans;
    const int m = strtol(argv[4], NULL, 10);
    const int n = strtol(argv[5], NULL, 10);
    const int k = strtol(argv[6], NULL, 10);
    const double alpha = strtod(argv[7], NULL);
    const double beta = strtod(argv[8], NULL);
    const int iteration = strtol(argv[9], NULL, 10);
    const int lda = (TransA == CblasTrans) != (layout == CblasRowMajor) ? k : m;
    const int ldb = (TransB == CblasTrans) != (layout == CblasRowMajor) ? n : k;
    const int ldc = layout == CblasRowMajor ? n : m;

    printf("---------------------------------------\n");
    printf("Kernel:  %s\n", KERNEL);
    printf("---------------------------------------\n");
    printf("Layout:  %s\n", (layout == CblasRowMajor) ? "Row" : "Column");
    printf("TransA:  %s\n", (TransA == CblasTrans) ? "Yes" : "No");
    printf("TransB:  %s\n", (TransB == CblasTrans) ? "Yes" : "No");
    printf("M:       %d\n", m);
    printf("N:       %d\n", n);
    printf("K:       %d\n", k);
    printf("alpha:   %.3lf\n", alpha);
    printf("beta:    %.3lf\n", beta);
    printf("---------------------------------------\n");

    double *A;
    double *B;
    double *C;
#ifdef VERIFY
    double *D;
#endif
    ALLOCATE(A, sizeof(double) * m * k);
    ALLOCATE(B, sizeof(double) * k * n);
    ALLOCATE(C, sizeof(double) * m * n);
#ifdef VERIFY
    ALLOCATE(D, sizeof(double) * m * n);
#endif

    set_data(A, m * k, 100, 0.0, 2.0);
    set_data(B, k * n, 200, 0.0, 2.0);
    set_data(C, m * n, 300, 0.0, 2.0);
#ifdef VERIFY
    memcpy(D, C, sizeof(double) * m * n);
#endif

    printf("case    duration(sec)    gflops(GFLOPS)\n");

    double total_duration = 0;
    double total_gflops = 0;
    double difference = 0;
    for (int i = 0; i < iteration; ++i) {
        const double start_time = omp_get_wtime();
        call_dgemm(layout, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        const double end_time = omp_get_wtime();
        const double duration = end_time - start_time;
        const double gflops = 2.0e-9 * m * n * k / duration;

        total_duration += duration;
        total_gflops += gflops;

        printf("%3d)%17.3lf%18.3lf\n", i + 1, duration, gflops);

#ifdef VERIFY
        if (i == 0) {
            cblas_dgemm(layout, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta, D, ldc);
            cblas_daxpy(m * n, -1.0, C, 1, D, 1);
            difference = cblas_dnrm2(m * n, D, 1);
        }
#endif
    }

    printf("---------------------------------------\n");
    const double avg_duration = total_duration / iteration;
    const double avg_gflops = total_gflops / iteration;
    printf("avg)%17.3lf%18.3lf\n", avg_duration, avg_gflops);

    FREE(A);
    FREE(B);
    FREE(C);
#ifdef VERIFY
    {
        int is_correct = difference < 0.0001;
        if (!is_correct) {
            printf("difference: %.4lf\n", difference);
            printf("---------------------------------------\n");
            return 1;
        }
        FREE(D);
    }
#endif

    printf("---------------------------------------\n");

    return 0;
}

static void set_data(double *matrix, int size, int seed, double min_value,
                     double max_value) {
#pragma omp parallel
    {
        unsigned value = (omp_get_thread_num() * 103 + 10581) * seed;
        const unsigned int mul = 192499;
        const unsigned int add = 6837199;

#pragma omp for
        for (int i = 0; i < size; ++i) {
            value = value * mul + add;
            matrix[i] = (double)value / (unsigned int)(-1) * (max_value - min_value) + min_value;
        }
    }
}
