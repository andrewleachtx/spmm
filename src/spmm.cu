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
__global__ void spmm_kernel(size_t* i, size_t* j, float* k, size_t nnz,
                            float* B, float* C) {}

int main() {
    // these guys are flattened, accessing B[r][c] should be B[r * N + c] and C[r][c] should also be C[r * N + c]
    fun::gpu_array<float> B(K * N);
    fun::gpu_array<float> C(M * N);

    // TODO(O): populate i, j, k, A, B, C and get nnz?
    size_t nnz = 0;
    fun::gpu_array<size_t> i(M);
    fun::gpu_array<size_t> j(nnz);
    fun::gpu_array<float> k(nnz);

    // send to GPU
    fun::to_device_all(i, j, k, B);

    spmm_kernel<<<67, 67>>>(i.device_ptr(), j.device_ptr(), k.device_ptr(), nnz,
                            B.device_ptr(), C.device_ptr());

    C.to_host();

    printf("hello worlds\n");
    test();
    // TODO: compare to some other library or check output i guess
}
