#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//access pointers like csr.i.data()
struct CSR {
    std::vector<int> i;
    std::vector<int> j;
    std::vector<float> k;
};

int test();

Eigen::SparseMatrix<float, Eigen::RowMajor> dense_to_sparse(Eigen::MatrixXf A);
CSR sparse_to_CSR(Eigen::SparseMatrix<float, Eigen::RowMajor> A_sparse);

RowMatrixXf random_dense(int n, bool asFloat = false);
Eigen::SparseMatrix<float, Eigen::RowMajor> random_sparse(int n, int nnz = -1,
                                                          bool asFloat = false);
