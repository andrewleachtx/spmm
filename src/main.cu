#include "gpu_array.hpp"
#include "sparse_csr.h"
#include "spmm.cuh"
#include <cassert>
#include <cusparse.h>
#include <iostream>
#include <random>
#include <string>

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static bool ends_with_mtx(const std::string& s) {
    return s.size() >= 4 && s.compare(s.size() - 4, 4, ".mtx") == 0;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        printf("Usage: ./spmm <n> (random nxn sparse matrix)\n");
        printf(
            "./spmm <file.mtx> [ncols in B] (load from Matrix Market file)\n");
        return 1;
    }

    CSR A_csr;
    int M_int, K_int;
    size_t N;

    if (!ends_with_mtx(argv[1])) {
        // Random mode: ./spmm <n>
        const int n = std::stoi(argv[1]);
        const size_t nnz = n * 100;

        Eigen::SparseMatrix<float, Eigen::RowMajor> A_eigen =
            random_sparse(n, nnz, true);

        M_int = static_cast<int>(A_eigen.rows());
        K_int = static_cast<int>(A_eigen.cols());
        N = static_cast<size_t>(n);

        A_csr = sparse_to_CSR(A_eigen);
    }
    else {
        // File mode: ./spmm <file.mtx> [feat_size]
        A_csr = load_mtx(argv[1], M_int, K_int);
        N = (argc >= 3) ? static_cast<size_t>(std::stoi(argv[2])) : 128;
    }

    const size_t M = static_cast<size_t>(M_int);
    const size_t K = static_cast<size_t>(K_int);

    // Generate random dense matrix B of size K x N
    RowMatrixXf B_eigen(K, N);
    {
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> dist(0, 9);
        for (size_t r = 0; r < K; ++r)
            for (size_t c = 0; c < N; ++c)
                B_eigen(r, c) = dist(gen);
    }

    // Allocate and populate device versions of A, B, C
    const size_t actual_nnz = A_csr.j.size();
    fun::gpu_array<int> i { A_csr.i.data(), M + 1 };
    fun::gpu_array<int> j { A_csr.j.data(), actual_nnz };
    fun::gpu_array<float> k { A_csr.k.data(), actual_nnz };
    fun::gpu_array<float> B { B_eigen.data(), K * N };
    fun::gpu_array<float> C { M * N };

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    spmm_kernel<<<blks_per_grid, threads_per_blk>>>(
        i.device_ptr(), j.device_ptr(), k.device_ptr(), B.device_ptr(),
        C.device_ptr(), M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ours_ms = 0;
    cudaEventElapsedTime(&time_ours_ms, start, stop);
    std::cout << "Ours: " << time_ours_ms << " ms" << std::endl;

    C.to_host();

    Eigen::Map<RowMatrixXf> C_eigen(&C[0], M, N);

    // cuSPARSE SPMM //
    fun::gpu_array<float> C_cusp { M * N };
    C_cusp.to_device();

    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);

    // Create sparse matrix descriptor (CSR)
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, M, K, actual_nnz, i.device_ptr(), j.device_ptr(),
                      k.device_ptr(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Create dense matrix descriptors (row-major)
    cusparseDnMatDescr_t matB, matC;
    cusparseCreateDnMat(&matB, K, N, N, B.device_ptr(), CUDA_R_32F,
                        CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, M, N, N, C_cusp.device_ptr(), CUDA_R_32F,
                        CUSPARSE_ORDER_ROW);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Query workspace size
    size_t buffer_size = 0;
    cusparseSpMM_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                            matB, &beta, matC, CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);

    void* d_buffer = nullptr;
    cudaMalloc(&d_buffer, buffer_size);

    cudaEventRecord(start);
    cusparseSpMM(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta,
                 matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buffer);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_cusparse = 0;
    cudaEventElapsedTime(&ms_cusparse, start, stop);
    std::cout << "cuSPARSE: " << ms_cusparse << " ms" << std::endl;

    C_cusp.to_host();

    // Cleanup
    cudaFree(d_buffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(cusparse_handle);

    Eigen::Map<RowMatrixXf> C_cusp_eigen(&C_cusp[0], M, N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
