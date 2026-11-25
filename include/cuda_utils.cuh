#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct DeviceLimits {
    cudaDeviceProp prop{};
    int max_warps_per_sm = 0;
};

inline DeviceLimits query_device_limits(int device = 0) {
    DeviceLimits limits{};
    CUDA_CHECK(cudaGetDeviceProperties(&limits.prop, device));
    limits.max_warps_per_sm =
        limits.prop.maxThreadsPerMultiProcessor / limits.prop.warpSize;
    return limits;
}

inline void print_device_limits(const DeviceLimits& limits) {
    const auto& p = limits.prop;
    std::printf("=== GPU & SM Hardware Limits ===\n");
    std::printf("GPU Name                : %s\n", p.name);
    std::printf("Compute Capability      : %d.%d\n", p.major, p.minor);
    std::printf("SM Count                : %d\n", p.multiProcessorCount);
    std::printf("Warp Size               : %d\n", p.warpSize);
    std::printf("Max Threads / SM        : %d\n", p.maxThreadsPerMultiProcessor);
    std::printf("Max Warps / SM          : %d\n", limits.max_warps_per_sm);
    std::printf("Max Blocks / SM         : %d\n", p.maxBlocksPerMultiProcessor);
    std::printf("Max Threads / Block     : %d\n", p.maxThreadsPerBlock);
    std::printf("Shared Memory / SM      : %zu bytes (%.1f KB)\n",
                p.sharedMemPerMultiprocessor,
                static_cast<double>(p.sharedMemPerMultiprocessor) / 1024.0);
    std::printf("Shared Memory / Block   : %zu bytes (%.1f KB)\n",
                p.sharedMemPerBlock,
                static_cast<double>(p.sharedMemPerBlock) / 1024.0);
    std::printf("Registers / SM          : %d\n", p.regsPerMultiprocessor);
    std::printf("================================\n\n");
}

inline std::string compute_limiting_resource(const DeviceLimits& limits,
                                             const cudaFuncAttributes& attr,
                                             int block_size,
                                             size_t dyn_shared_bytes,
                                             int active_blocks_per_sm) {
    const auto& p = limits.prop;
    const int limit_threads = p.maxThreadsPerMultiProcessor / block_size;
    const int limit_blocks = p.maxBlocksPerMultiProcessor;
    const int limit_regs =
        (attr.numRegs > 0)
            ? (p.regsPerMultiprocessor / (attr.numRegs * block_size))
            : limit_blocks;

    const size_t total_smem = attr.sharedSizeBytes + dyn_shared_bytes;
    const int limit_smem =
        (total_smem > 0)
            ? static_cast<int>(p.sharedMemPerMultiprocessor / total_smem)
            : limit_blocks;

    const int min_limit = std::min(
        {limit_threads, limit_blocks, limit_regs, limit_smem});

    // If hardware granularity/reserved shared memory reduces active blocks below
    // the raw byte division when shared memory is already the tightest bound,
    // shared_mem_per_sm is the active bottleneck.
    if (active_blocks_per_sm > 0 && active_blocks_per_sm < min_limit &&
        total_smem > 0 && limit_smem == min_limit) {
        return "shared_mem_per_sm";
    }

    std::string label;
    auto append_label = [&](const char* name) {
        if (!label.empty()) label += "+";
        label += name;
    };

    if (limit_smem == min_limit && total_smem > 0) append_label("shared_mem_per_sm");
    if (limit_threads == min_limit) append_label("threads_per_sm");
    if (limit_blocks == min_limit) append_label("blocks_per_sm");
    if (limit_regs == min_limit) append_label("registers_per_sm");

    return label.empty() ? "unknown" : label;
}
