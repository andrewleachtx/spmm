#include "sparse_csr.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

int test() {
    Eigen::MatrixXf A(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    std::cout << A << std::endl;

    CSR test_out = dense_to_sparse(A);
    for (int x : test_out.i) {
        std::cout << x << " ";
    }
    for (int x : test_out.j) {
        std::cout << x << " ";
    }
    for (int x : test_out.k) {
        std::cout << x << " ";
    }
    return 0;
}

CSR dense_to_sparse(Eigen::MatrixXf A) {
    Eigen::SparseMatrix<float, Eigen::RowMajor> A_sparse = A.sparseView();
    A_sparse.makeCompressed();
    CSR output;
    output.i.assign(A_sparse.outerIndexPtr(),
                    A_sparse.outerIndexPtr() + A_sparse.rows() + 1);

    output.j.assign(A_sparse.innerIndexPtr(),
                    A_sparse.innerIndexPtr() + A_sparse.nonZeros());

    output.k.assign(A_sparse.valuePtr(),
                    A_sparse.valuePtr() + A_sparse.nonZeros());
    return output;
}
