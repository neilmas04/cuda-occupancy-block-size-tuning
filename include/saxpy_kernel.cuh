#pragma once

#include <cuda_runtime.h>

__global__ void saxpy_kernel(int n, float a, const float* __restrict__ x,
                             float* __restrict__ y, size_t dyn_shared_bytes) {
    extern __shared__ float s_buf[];

    float smem_zero = 0.0f;
    if (dyn_shared_bytes >= sizeof(float)) {
        const int num_floats = static_cast<int>(dyn_shared_bytes / sizeof(float));
        const int slot = threadIdx.x % num_floats;
        s_buf[slot] = 0.0f;
        __syncthreads();
        smem_zero = s_buf[slot];
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx] + smem_zero;
    }
}
