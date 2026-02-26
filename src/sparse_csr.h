#pragma once
#include <vector>
#include <Eigen/Dense>

//access pointers like csr.i.data()
struct CSR {
    std::vector<int> i;
    std::vector<int> j;
    std::vector<float> k;
};

int test();

CSR dense_to_sparse(Eigen::MatrixXf A);