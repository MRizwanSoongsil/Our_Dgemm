#!/bin/bash

set -e

module purge
module load craype-mic-knl intel/oneapi_21.2 impi/oneapi_21.2

make all
