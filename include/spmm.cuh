#pragma once
#include <cstddef>

#define WARP_SZ 32

__global__ void spmm_kernel(const int* row_ptr, const int* col_ptr,
                            const float* val_ptr, const float* B, float* C,
                            const size_t M, const size_t N);
