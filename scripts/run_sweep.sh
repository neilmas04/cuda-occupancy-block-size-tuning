#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda/bin:${PATH}"

mkdir -p build results plots docs
nvcc -O3 -arch=sm_89 --ptxas-options=-v -Iinclude src/saxpy.cu -o build/saxpy_sweep
./build/saxpy_sweep 33554432 results/saxpy_dyn_shared_sweep.csv
