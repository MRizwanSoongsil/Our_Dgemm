#ifdef USE_HBWMALLOC
#include <hbwmalloc.h>
#define ALLOCATE(ptr, size) hbw_posix_memalign((void**)&(ptr), 64, (size))
#define FREE(ptr) hbw_free(ptr)
#else
#define ALLOCATE(ptr, size) ptr = malloc((size))
#define FREE(ptr) free(ptr)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include <mkl_blacs.h>
#include <mkl_scalapack.h> 

#ifndef KERNEL
#define KERNEL "UNKNOWN"
#endif

static void set_data(double *matrix, int size, int seed, double min_value, double max_value);

int main(int argc, char **argv) {
    const int ZERO = 0, ONE = 1, MINUS_ONE = -1;

    if (argc != 6) {
        fprintf(stderr, "Usage : %s m mb nprow npcol iteration\n", argv[0]);
        return 1;
    }

    int m = strtol(argv[1], NULL, 10);
    int mb = strtol(argv[2], NULL, 10);
    int nprow = strtol(argv[3], NULL, 10);
    int npcol = strtol(argv[4], NULL, 10);
    int iteration = strtol(argv[5], NULL, 10);

    int myrank_mpi, nprocs_mpi;
    blacs_pinfo_(&myrank_mpi, &nprocs_mpi);

    int ictxt, myrow, mycol;
    blacs_get_(&MINUS_ONE, &ZERO, &ictxt);
    blacs_gridinit_(&ictxt, "R", &nprow, &npcol);
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);

    int rA = numroc_(&m, &mb, &myrow, &ZERO, &nprow);
    int cA = numroc_(&m, &mb, &mycol, &ZERO, &npcol);

    int descA[9], info;
    descinit_(descA, &m, &m, &mb, &mb, &ZERO, &ZERO, &ictxt, &rA, &info);

    double *A;
    double *tau;
    ALLOCATE(A, sizeof(double) * rA * cA);
    ALLOCATE(tau, sizeof(double) * cA);

    set_data(A, rA * cA, 100, -1.0, +1.0);

    double d_lwork;
    pdgeqrf_(&m, &m, A, &ONE, &ONE, descA, tau, &d_lwork, &MINUS_ONE, &info);
    int i_lwork = (int)d_lwork;

    double *work;
    ALLOCATE(work, sizeof(double) * i_lwork);

    if (myrow == 0 && mycol == 0) {
        printf("---------------------------------------\n");
        printf("Kernel:  %s\n", KERNEL);
        printf("---------------------------------------\n");
        printf("M:       %d\n", m);
        printf("MB:      %d\n", mb);
        printf("nprow:   %d\n", nprow);
        printf("npcol:   %d\n", npcol);
        printf("---------------------------------------\n");
        printf("case    duration(sec)    gflops(GFLOPS)\n");
    }

    for(int i=0; i<iteration; ++i) {
//      Cblacs_barrier(ictxt, "All");
        double start=MPI_Wtime();
        pdgeqrf_(&m, &m, A, &ONE, &ONE, descA, tau, work, &i_lwork, &info);
//      Cblacs_barrier(ictxt, "All");
        double end=MPI_Wtime();
        double duration = end - start; 
        double gflops = 4.0 * m * m * m / 3.0 * 1.0e-9 / duration;

        if (myrow == 0 && mycol == 0) {
            printf("%3d)%17.3lf%18.3lf\n", i + 1, duration, gflops);
        }
    }

    if (myrow == 0 && mycol == 0) {
        printf("---------------------------------------\n");
    }

    FREE(A);
    FREE(tau);
    FREE(work);
//  Cblacs_gridexit(0);
    MPI_Finalize();
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
            matrix[i] =
                (double)value / (unsigned int)(-1) * (max_value - min_value) +
                min_value;
        }
    }
}
