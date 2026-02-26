#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

//access pointers like csr.i.data()
struct CSR {
    std::vector<int> i;
    std::vector<int> j;
    std::vector<float> k;
};

int test();

Eigen::SparseMatrix<float, Eigen::RowMajor> dense_to_sparse(Eigen::MatrixXf A);
CSR sparse_to_CSR(Eigen::SparseMatrix<float, Eigen::RowMajor> A_sparse);

Eigen::MatrixXf random_dense(int n);
Eigen::SparseMatrix<float, Eigen::RowMajor> random_sparse(int n);
