#include "sparse_csr.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <random>

int test() {
    Eigen::MatrixXf A(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    std::cout << A << std::endl;

    CSR test_out = sparse_to_CSR(random_sparse(4));
    for (int x : test_out.i) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    for (int x : test_out.j) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    for (int x : test_out.k) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    return 0;
}

Eigen::SparseMatrix<float, Eigen::RowMajor> dense_to_sparse(Eigen::MatrixXf A) {
    Eigen::SparseMatrix<float, Eigen::RowMajor> A_sparse = A.sparseView();
    A_sparse.makeCompressed();
    return A_sparse;
}

CSR sparse_to_CSR(Eigen::SparseMatrix<float, Eigen::RowMajor> A_sparse) {
    CSR output;
    output.i.assign(A_sparse.outerIndexPtr(),
                    A_sparse.outerIndexPtr() + A_sparse.rows() + 1);

    output.j.assign(A_sparse.innerIndexPtr(),
                    A_sparse.innerIndexPtr() + A_sparse.nonZeros());

    output.k.assign(A_sparse.valuePtr(),
                    A_sparse.valuePtr() + A_sparse.nonZeros());
    return output;
}

// TODO (O) helpers to make big sparse and dense matrices

// Random matrix of size nxn with uniformly distributed random values in [0, 9]
Eigen::MatrixXf random_dense(int n) {
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(n, n); 
    
    A = A*5+Eigen::MatrixXf::Constant(n, n, 5.0);
    A = A.array().floor().matrix();
    std::cout << A << std::endl;
    return A;
}

Eigen::SparseMatrix<float, Eigen::RowMajor> random_sparse(int n) {

    std::mt19937 gen;
    std::uniform_int_distribution<int> index_dist(0, n - 1);
    std::uniform_int_distribution<int> value_dist(0, 9);
    int nnz = n;
    std::vector<Eigen::Triplet<int>> triplets;
    triplets.reserve(nnz);
    for (int x = 0; x < nnz; x++) {
        int i = index_dist(gen);
        int j = index_dist(gen);
        int k = value_dist(gen);
        printf("%d\n", k);
        
        triplets.emplace_back(i, j, k);
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end(), [](float /*old*/, float new_val) {
        return new_val;   // overwrite instead of sum
    });

    A.makeCompressed();
    return A;
}