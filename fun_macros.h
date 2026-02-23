#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define cudaTry(ans)                                                           \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }

__inline __host__ void gpuAssert(cudaError_t code, const char* file, int line,
                                 bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cudaTry ASSERTION FAILED: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
