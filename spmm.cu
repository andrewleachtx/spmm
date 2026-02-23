#include "gpu_array.hpp"

constexpr int M = 1 << 10;
constexpr int K = 1 << 10;
constexpr int N = 1 << 10;

/*
Given an m-by-k sparse matrix A and a k-by-n dense matrix B, SpMM computes
an m-by-n dense matrix C = AB.

TODO: B, C are dense, so we can just use a float* but A needs a CSR format
    - A: i[], j[], k[] can be size_t*, size_t*, float*

*/
__global__ void spmm_kernel(float* B, float* C) {}

int main() {
    // these guys are flattened, accessing B[r][c] should be B[r * N + c] and C[r][c] should also be C[r * N + c]
    fun::gpu_array<float, K * N> B;
    fun::gpu_array<float, M * N> C;

    // TODO: populate A, B, C

    B.to_device();
    C.to_device();

    spmm_kernel<<<67, 67>>>(B.device_ptr(), C.device_ptr());

    B.to_host();
    C.to_host();
}
