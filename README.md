# Our_Dgemm

This repository contains the source code accompanying the paper titled "Revisiting the Performance Optimization of QR
Factorization on Intel KNL and SKL
Multiprocessors" by Rizwan  et al. 

## Prerequisites

Before running the code, you must have installed the following libraries installed on your system:
- memkind
- ScaLAPACK version 2.2.0
- oneMKL
For benchmarking with BLIS and OpenBlas, following liberaries are also required
- BLIS
- OpenBLAS

These libraries are essential for the proper execution of the program. Please follow the installation instructions provided by each library's official documentation.

## Environment Setup

To ensure the code runs correctly, you need to set the following environment variables. Replace `[INSTALLATION_PREFIX]` with the path where your libraries are installed and `<memkind library installation path>` with the actual installation path of the memkind library.

```bash
export LIBROOT=[INSTALLATION_PREFIX]/lib
export MEMKINDROOT=<memkind library installation path>
export BLISROOT=${LIBROOT}/blis
export OPENBLASROOT=${LIBROOT}/OpenBLAS
export SCALAPACKROOT=${LIBROOT}/scalapack-2.2.0

export CPATH=$CPATH:${MEMKINDROOT}/include:${MKLROOT}/include:${BLISROOT}/include
export LIBRARY_PATH=$LIBRARY_PATH:${MEMKINDROOT}/lib:${MKLROOT}/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MEMKINDROOT}/lib:${MKLROOT}/lib/intel64
