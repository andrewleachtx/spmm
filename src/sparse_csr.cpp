#include "sparse_csr.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

extern "C" {
#include "mmio.h"
}

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
RowMatrixXf random_dense(int n, bool asFloat) {
    std::mt19937 gen(0);
    RowMatrixXf A(n, n);

    if (asFloat) {
        std::uniform_real_distribution<float> dist(0, 9);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A(i, j) = static_cast<float>(dist(gen));
    }
    else {
        std::uniform_int_distribution<int> dist(0, 9);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A(i, j) = static_cast<float>(dist(gen));
    }

    return A;
}

Eigen::SparseMatrix<float, Eigen::RowMajor> random_sparse(int n, int nnz,
                                                          bool asFloat) {
    if (nnz < 0 || nnz > n * n) {
        nnz = n;
    }

    // std::mt19937 gen(std::random_device{}());
    std::mt19937 gen(0);

    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(nnz);
    std::uniform_int_distribution<int> index_dist(0, n - 1);
    if (asFloat) {
        std::uniform_real_distribution<float> value_dist(0, 9);
        for (int x = 0; x < nnz; x++) {
            int i = index_dist(gen);
            int j = index_dist(gen);
            float k = value_dist(gen);
            triplets.emplace_back(i, j, k);
        }
    }
    else {
        std::uniform_int_distribution<int> value_dist(0, 9);
        for (int x = 0; x < nnz; x++) {
            int i = index_dist(gen);
            int j = index_dist(gen);
            float k = static_cast<float>(value_dist(gen));
            triplets.emplace_back(i, j, k);
        }
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end(),
                      [](float /*old*/, float new_val) {
                          return new_val; // overwrite instead of sum
                      });

    A.makeCompressed();
    return A;
}

CSR load_mtx(const std::string& path, int& M, int& N) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "Error: could not open file %s\n", path.c_str());
        exit(1);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Error: could not read Matrix Market banner in %s\n",
                path.c_str());
        fclose(f);
        exit(1);
    }

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        fprintf(stderr, "Error: only sparse matrices are supported (got %s)\n",
                mm_typecode_to_str(matcode));
        fclose(f);
        exit(1);
    }

    if (mm_is_complex(matcode)) {
        fprintf(stderr, "Error: complex matrices are not supported\n");
        fclose(f);
        exit(1);
    }

    int nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Error: could not read matrix size\n");
        fclose(f);
        exit(1);
    }

    std::cout << "Loading " << path << ": " << M << "x" << N << ", " << nz
              << " nonzeros";
    if (mm_is_symmetric(matcode) || mm_is_skew(matcode))
        std::cout << " (symmetric)";
    if (mm_is_pattern(matcode))
        std::cout << " (pattern)";
    std::cout << std::endl;

    bool is_symmetric = mm_is_symmetric(matcode) || mm_is_skew(matcode);
    bool is_pattern = mm_is_pattern(matcode);
    bool is_integer = mm_is_integer(matcode);

    // Reserve extra space for symmetric matrices (off-diag entries get mirrored)
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(is_symmetric ? nz * 2 : nz);

    for (int idx = 0; idx < nz; idx++) {
        int row, col;
        double val = 1.0;

        if (is_pattern) {
            if (fscanf(f, "%d %d", &row, &col) != 2) {
                fprintf(stderr, "Error: premature EOF at entry %d\n", idx);
                fclose(f);
                exit(1);
            }
        }
        else if (is_integer) {
            int ival;
            if (fscanf(f, "%d %d %d", &row, &col, &ival) != 3) {
                fprintf(stderr, "Error: premature EOF at entry %d\n", idx);
                fclose(f);
                exit(1);
            }
            val = static_cast<double>(ival);
        }
        else {
            if (fscanf(f, "%d %d %lg", &row, &col, &val) != 3) {
                fprintf(stderr, "Error: premature EOF at entry %d\n", idx);
                fclose(f);
                exit(1);
            }
        }

        // Matrix Market is 1-based
        row--;
        col--;

        triplets.emplace_back(row, col, static_cast<float>(val));

        if (is_symmetric && row != col) {
            float sym_val = mm_is_skew(matcode) ? -static_cast<float>(val)
                                                : static_cast<float>(val);
            triplets.emplace_back(col, row, sym_val);
        }
    }

    fclose(f);

    Eigen::SparseMatrix<float, Eigen::RowMajor> A(M, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    std::cout << "Loaded: " << A.rows() << "x" << A.cols() << ", "
              << A.nonZeros() << " nonzeros (after expansion)" << std::endl;

    return sparse_to_CSR(A);
}
