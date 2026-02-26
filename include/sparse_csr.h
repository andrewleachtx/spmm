#pragma once
#include <Eigen/Dense>
#include <vector>

//access pointers like csr.i.data()
struct CSR {
    std::vector<int> i;
    std::vector<int> j;
    std::vector<float> k;
};

int test();

CSR dense_to_sparse(Eigen::MatrixXf A);
