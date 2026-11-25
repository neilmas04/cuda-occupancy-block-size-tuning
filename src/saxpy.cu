#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "cuda_utils.cuh"
#include "saxpy_kernel.cuh"

struct SweepResult {
    int n = 0;
    int block_size = 0;
    int grid_size = 0;
    size_t dyn_shared_bytes = 0;
    int active_blocks_per_sm = 0;
    double theoretical_occupancy_pct = 0.0;
    std::string limiting_resource;
    double avg_time_ms = 0.0;
    double bandwidth_gb_s = 0.0;
    float max_err = 0.0f;
};

int main(int argc, char** argv) {
    int n = 33554432;  // 2^25 elements (128 MB per array, 384 MB total traffic)
    std::string csv_path = "results/saxpy_dyn_shared_sweep.csv";
    const int warmup_iters = 10;
    const int bench_iters = 100;

    if (argc >= 2) {
        n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        csv_path = argv[2];
    }
    if (n <= 0) {
        std::fprintf(stderr, "Usage: %s [N] [csv_path]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    DeviceLimits limits = query_device_limits(device);
    print_device_limits(limits);

    cudaFuncAttributes func_attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&func_attr, saxpy_kernel));
    std::printf("Kernel Attributes: regs/thread = %d, static_smem = %zu B\n\n",
                func_attr.numRegs, func_attr.sharedSizeBytes);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const float a = 2.0f;

    std::vector<float> h_x(n);
    std::vector<float> h_y_init(n);
    std::vector<float> h_y_ref(n);
    std::vector<float> h_y_gpu(n);

    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i % 100) * 0.01f;
        h_y_init[i] = static_cast<float>((i * 3) % 100) * 0.02f;
        h_y_ref[i] = a * h_x[i] + h_y_init[i];
    }

    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    const std::vector<size_t> dyn_shared_configs = {0, 16384, 32768};

    std::vector<SweepResult> results;
    results.reserve(block_sizes.size() * dyn_shared_configs.size());

    std::printf("Benchmark Config: N = %d, Warm-up = %d, Timed Iters = %d\n\n",
                n, warmup_iters, bench_iters);
    std::printf(
        "%-9s %-10s %-7s %-10s %-34s %-12s %-11s %-10s\n",
        "BlockSize", "DynSmem(B)", "Blk/SM", "TheoOcc(%)", "LimitingResource",
        "AvgTime(ms)", "BW(GB/s)", "MaxErr");
    std::printf(
        "----------------------------------------------------------------------------------------------------------------\n");

    for (size_t dyn_shared_bytes : dyn_shared_configs) {
        if (dyn_shared_bytes + func_attr.sharedSizeBytes >
            limits.prop.sharedMemPerBlock) {
            std::fprintf(
                stderr,
                "Skipping dyn_shared_bytes=%zu (exceeds sharedMemPerBlock=%zu)\n",
                dyn_shared_bytes, limits.prop.sharedMemPerBlock);
            continue;
        }

        for (int block_size : block_sizes) {
            if (block_size > limits.prop.maxThreadsPerBlock) {
                continue;
            }

            const int grid_size = (n + block_size - 1) / block_size;

            // 1. Reset d_y and verify GPU output against CPU reference before timing
            CUDA_CHECK(cudaMemcpy(d_y, h_y_init.data(), bytes,
                                  cudaMemcpyHostToDevice));
            saxpy_kernel<<<grid_size, block_size, dyn_shared_bytes>>>(
                n, a, d_x, d_y, dyn_shared_bytes);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, bytes,
                                  cudaMemcpyDeviceToHost));
            float max_err = 0.0f;
            for (int i = 0; i < n; ++i) {
                float err = std::fabs(h_y_gpu[i] - h_y_ref[i]);
                if (err > max_err) {
                    max_err = err;
                }
            }
            if (max_err > 1e-5f) {
                std::fprintf(
                    stderr,
                    "Verification FAILED for block_size=%d, dyn_shared=%zu! max_err=%e\n",
                    block_size, dyn_shared_bytes, max_err);
                CUDA_CHECK(cudaFree(d_x));
                CUDA_CHECK(cudaFree(d_y));
                return EXIT_FAILURE;
            }

            // 2. Warm-up runs
            for (int i = 0; i < warmup_iters; ++i) {
                saxpy_kernel<<<grid_size, block_size, dyn_shared_bytes>>>(
                    n, a, d_x, d_y, dyn_shared_bytes);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // 3. Timed kernel runs (device-only, excluding H2D/D2H transfers)
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < bench_iters; ++i) {
                saxpy_kernel<<<grid_size, block_size, dyn_shared_bytes>>>(
                    n, a, d_x, d_y, dyn_shared_bytes);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            const double avg_time_ms =
                static_cast<double>(total_ms) / bench_iters;

            // SAXPY traffic: read x (4B) + read y (4B) + write y (4B) = 12B per element
            const double total_bytes = 3.0 * static_cast<double>(bytes);
            const double bandwidth_gb_s =
                total_bytes / (avg_time_ms * 1e-3) / 1e9;

            int active_blocks_per_sm = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_sm, saxpy_kernel, block_size,
                dyn_shared_bytes));

            const double theo_occ_pct =
                100.0 * static_cast<double>(active_blocks_per_sm * block_size) /
                static_cast<double>(limits.prop.maxThreadsPerMultiProcessor);

            const std::string limiting_resource = compute_limiting_resource(
                limits, func_attr, block_size, dyn_shared_bytes,
                active_blocks_per_sm);

            SweepResult r{};
            r.n = n;
            r.block_size = block_size;
            r.grid_size = grid_size;
            r.dyn_shared_bytes = dyn_shared_bytes;
            r.active_blocks_per_sm = active_blocks_per_sm;
            r.theoretical_occupancy_pct = theo_occ_pct;
            r.limiting_resource = limiting_resource;
            r.avg_time_ms = avg_time_ms;
            r.bandwidth_gb_s = bandwidth_gb_s;
            r.max_err = max_err;
            results.push_back(r);

            std::printf(
                "%-9d %-10zu %-7d %-10.2f %-34s %-12.6f %-11.2f %-10.3e\n",
                r.block_size, r.dyn_shared_bytes, r.active_blocks_per_sm,
                r.theoretical_occupancy_pct, r.limiting_resource.c_str(),
                r.avg_time_ms, r.bandwidth_gb_s, r.max_err);
        }
    }

    // Write CSV
    std::ofstream ofs(csv_path);
    if (!ofs.is_open()) {
        std::fprintf(stderr, "Failed to open CSV file for writing: %s\n",
                     csv_path.c_str());
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        return EXIT_FAILURE;
    }
    ofs << "n,block_size,grid_size,dyn_shared_bytes,active_blocks_per_sm,"
           "theoretical_occupancy_pct,limiting_resource,avg_time_ms,"
           "bandwidth_gb_s,max_err\n";
    for (const auto& r : results) {
        ofs << r.n << "," << r.block_size << "," << r.grid_size << ","
            << r.dyn_shared_bytes << "," << r.active_blocks_per_sm << ","
            << r.theoretical_occupancy_pct << "," << r.limiting_resource << ","
            << r.avg_time_ms << "," << r.bandwidth_gb_s << "," << r.max_err
            << "\n";
    }
    ofs.close();
    std::printf("\nSaved Phase 3 sweep results to: %s\n", csv_path.c_str());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return EXIT_SUCCESS;
}
