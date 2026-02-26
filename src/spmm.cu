#include "gpu_array.hpp"
#include "sparse_csr.h"

constexpr int M = 1 << 10;
constexpr int K = 1 << 10;
constexpr int N = 1 << 10;

/*
Given an m-by-k sparse matrix A and a k-by-n dense matrix B, SpMM computes
an m-by-n dense matrix C = AB.

TODO: B, C are dense, so we can just use a float* but A needs a CSR format
    - A: i[], j[], k[] can be size_t*, size_t*, float*

*/
__global__ void spmm_kernel(float* B, float* C) {} //size_t* i, size_t* j, float* k, int nns, 

int main() {
    // these guys are flattened, accessing B[r][c] should be B[r * N + c] and C[r][c] should also be C[r * N + c]
    fun::gpu_array<float, K * N> B;
    fun::gpu_array<float, M * N> C;

    // fun::gpu_array<size_t, M> i;
    // fun::gpu_array<size_t, nns> j;
    // fun::gpu_array<float, nns> k;

    // TODO: populate A, B, C

    B.to_device();
    C.to_device();

    spmm_kernel<<<67, 67>>>(B.device_ptr(), C.device_ptr());

    B.to_host();
    C.to_host();

    printf("hello worlds");
    test();
    // TODO: compare to some other library or check output i guess
}
