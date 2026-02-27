#include "gpu_array.hpp"
#include "sparse_csr.h"
#include <cstring>
#include <iostream>

#define WARP_SZ 32

/*
Given an m-by-k sparse matrix A and a k-by-n dense matrix B, SpMM computes
an m-by-n dense matrix C = AB.
*/
__global__ void spmm_kernel(const int* row_ptr, const int* col_ptr,
                            const float* val_ptr, const size_t nnz,
                            const float* B, float* C, const size_t M,
                            const size_t N) {
    int lane = threadIdx.x % WARP_SZ;
    int per_blk_warp = threadIdx.x / WARP_SZ;
    int warps_per_blk = blockDim.x / WARP_SZ;

    const int row = blockIdx.x * warps_per_blk + per_blk_warp;
    const int col = blockIdx.y * WARP_SZ + lane;

    if (row >= M || col >= N) {
        return;
    }

    const int row_size = row_ptr[row + 1] - row_ptr[row];

    // not paper optimized yet
    float acc = 0.0f;
    const int x = row_ptr[row];
    for (int i = 0; i < row_size; i++) {
        int k = col_ptr[x + i];
        float a_val = val_ptr[x + i];
        float b_val = B[k * N + col];
        acc += a_val * b_val;
    }

    C[row * N + col] = acc;
}

int main() {

    // TODO(O): populate i, j, k, A, B, C and get nnz?

    // sample initializations
    Eigen::MatrixXf A(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    Eigen::MatrixXf B_dense(3, 3);
    B_dense << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    const int M = A.rows();
    const int K = A.cols();
    const int N = B_dense.cols();

    CSR A_csr = sparse_to_CSR(dense_to_sparse(A));
    int nnz = A_csr.j.size();

    // these guys are flattened, accessing B[r][c] should be B[r * N + c] and C[r][c] should also be C[r * N + c]
    fun::gpu_array<float> B(K * N);
    fun::gpu_array<float> C(M * N);

    fun::gpu_array<int> i(M + 1);
    fun::gpu_array<int> j(nnz);
    fun::gpu_array<float> k(nnz);

    std::memcpy(&i[0], A_csr.i.data(), (M + 1) * sizeof(int));
    std::memcpy(&j[0], A_csr.j.data(), nnz * sizeof(int));
    std::memcpy(&k[0], A_csr.k.data(), nnz * sizeof(float));

    // B_dense is column-major; kernel expects row-major
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        B_row = B_dense;
    std::memcpy(&B[0], B_row.data(), K * N * sizeof(float));

    /*
        - The amount of warps per block is arbitrary, really just used for threads per block (which is also a hyperparameter)
        - blockIdx.x selects a tile per row
            - then gridDim.x = ceil(M / warps_per_blk)
        - blockIdx.y selects a tile per 32 columns
            - then gridDim.y = ceil(N / 32)
            
        Together, the 2D grid gives us for each x block: multiple y blocks or chunks, of size cols / warp size
    */
    const int warps_per_blk = 16;
    const int threads_per_blk = warps_per_blk * WARP_SZ;
    dim3 blks_per_grid {
        // ceil(a / b) == (a + b - 1) / b
        static_cast<uint32_t>((M + warps_per_blk - 1) / warps_per_blk),
        static_cast<uint32_t>((N + WARP_SZ - 1) / WARP_SZ), 1
    };

    // send to GPU
    fun::to_device_all(i, j, k, B);

    spmm_kernel<<<blks_per_grid, threads_per_blk>>>(
        i.device_ptr(), j.device_ptr(), k.device_ptr(), nnz, B.device_ptr(),
        C.device_ptr(), M, N);

    C.to_host();

    // random_dense(4);
    // test();
    // TODO: compare to some other library or check output i guess
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        C_eigen(&C[0], M, N);

    Eigen::MatrixXf ref = A * B_dense;
    std::cout << "CUDA:\n" << C_eigen << "\n";
    std::cout << "Eigen:\n" << ref << "\n";
}
