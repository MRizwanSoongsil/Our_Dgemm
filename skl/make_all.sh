#!/bin/bash

set -e

module purge
module load craype-x86-skylake intel/oneapi_21.2 impi/oneapi_21.2 

make all
