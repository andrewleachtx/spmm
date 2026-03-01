#include "gpu_array.hpp"
#include "sparse_csr.h"
#include "spmm.cuh"
#include <iostream>

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int main() {

    // sample initializations
    Eigen::MatrixXf A_eigen(3, 3);
    A_eigen << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    RowMatrixXf B_eigen(3, 3);
    B_eigen << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    // TODO: can make the usage ./spmm <m> <k> <n> and use random generation with those preset sizes later
    const size_t M { static_cast<size_t>(A_eigen.rows()) };
    const size_t K { static_cast<size_t>(A_eigen.cols()) };
    const size_t N { static_cast<size_t>(B_eigen.cols()) };

    CSR A_csr = sparse_to_CSR(dense_to_sparse(A_eigen));
    const size_t nnz = A_csr.j.size();

    // Allocate and populate device versions of A, B, C
    fun::gpu_array<float> B { K * N }, C { M * N };
    fun::gpu_array<int> i { M + 1 }, j { nnz };
    fun::gpu_array<float> k { nnz };
    memcpy(&i[0], A_csr.i.data(), (M + 1) * sizeof(int));
    memcpy(&j[0], A_csr.j.data(), nnz * sizeof(int));
    memcpy(&k[0], A_csr.k.data(), nnz * sizeof(float));
    memcpy(&B[0], B_eigen.data(), K * N * sizeof(float));

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

    fun::to_device_all(i, j, k, B);

    spmm_kernel<<<blks_per_grid, threads_per_blk>>>(
        i.device_ptr(), j.device_ptr(), k.device_ptr(), B.device_ptr(),
        C.device_ptr(), M, N);

    C.to_host();

    // TODO add more thorough comparison/testing, we cant print a 1000x1000 i guess

    Eigen::Map<RowMatrixXf> C_eigen(&C[0], M, N);

    Eigen::MatrixXf ref = A_eigen * B_eigen;
    std::cout << "CUDA:\n" << C_eigen << "\n";
    std::cout << "Eigen:\n" << ref << "\n";
}
