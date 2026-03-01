#include "spmm.cuh"

/*
    Given an m-by-k sparse matrix A and a k-by-n dense matrix B, SpMM computes
    an m-by-n dense matrix C = AB.
*/
__global__ void spmm_kernel(const int* row_ptr, const int* col_ptr,
                            const float* val_ptr, const float* B, float* C,
                            const size_t M, const size_t N) {
    int lane = threadIdx.x % WARP_SZ;
    int per_blk_warp = threadIdx.x / WARP_SZ;
    int warps_per_blk = blockDim.x / WARP_SZ;

    const int row = blockIdx.x * warps_per_blk + per_blk_warp;
    const int col = blockIdx.y * WARP_SZ + lane;

    if (row >= M || col >= N) {
        return;
    }

    const int row_size = row_ptr[row + 1] - row_ptr[row];

    /*
        Each lane should take one of 32 '(k, a_v)' pairs from the current sparse row in A

        Then we know each lane needs that (k, a_v) in its sum somewhere, or rather its product a_v * b_v

        So we iterate in chunks of 32, let each lane increment its own counter by the current subset of the row/column
            - And increment its counter by the values that each other lane takes on
    */
    float acc = 0.0f;
    const int x = row_ptr[row];

    for (int col_offset = 0; col_offset < row_size; col_offset += 32) {
        int idx = x + lane + col_offset;

        // at the end some lanes exceed the row, so treat as zero contribution to the inner product
        int per_lane_k = 0;
        float per_lane_a = 0.0f;
        if (idx < x + row_size) {
            per_lane_k = col_ptr[idx];
            per_lane_a = val_ptr[idx];
        }

        const unsigned int all_mask = 0xffffffff;

        for (int src_lane = 0; src_lane < 32; src_lane++) {
            // __shfl_sync(mask, local var to share, thread/lane to store in return to)
            int k = __shfl_sync(all_mask, per_lane_k, src_lane);
            int a = __shfl_sync(all_mask, per_lane_a, src_lane);

            acc += a * B[k * N + col];
        }
    }

    C[row * N + col] = acc;
}
