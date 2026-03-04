#include "gpu_array.hpp"
#include "sparse_csr.h"
#include "spmm.cuh"
#include <cassert>
#include <cusparse.h>
#include <iostream>

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int main(int argc, char** argv) {
    if (argc != 2) {
        // TODO: can make the usage ./spmm <m> <k> <n> later, but square matrices is good for now
        printf("Usage: ./spmm <n>\n");
        return 1;
    }

    const int n = std::stoi(argv[1]);
    const size_t nnz = n * 100;

    Eigen::SparseMatrix<float, Eigen::RowMajor> A_eigen =
        random_sparse(n, nnz, true);
    RowMatrixXf B_eigen = random_dense(n, true);

    // Eigen MM (CPU) //
    // Eigen::MatrixXf reference = A_eigen * B_eigen;

    const size_t M { static_cast<size_t>(A_eigen.rows()) };
    const size_t K { static_cast<size_t>(A_eigen.cols()) };
    const size_t N { static_cast<size_t>(B_eigen.cols()) };

    CSR A_csr = sparse_to_CSR((A_eigen));

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
    // std::cout << "Blocks per grid x " << blks_per_grid.x << std::endl;
    // std::cout << "Blocks per grid y " << blks_per_grid.y << std::endl;

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

    // Eigen::MatrixXf diff = reference - C_eigen;
    // assert(diff.norm() < 1e-5f && "Ours norm != 0");

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
    // Eigen::MatrixXf diff_cusp = reference - C_cusp_eigen;
    // assert(diff_cusp.norm() < 1e-5f && "cuSPARSE norm != 0");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
