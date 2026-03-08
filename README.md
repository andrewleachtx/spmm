We are working in WSL2 for this project.

Be sure submodules are pulled (eigen).
```sh
git submodule update --init --recursive
```

# Building + Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

There is a simple `bench.sh`, or you can run manually with `./build/spmm <n>`.

# References
1. Design Principles for Sparse Matrix Multiplication on the GPU (https://arxiv.org/abs/1803.08601)
2. DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix Multiplication with Tensor Cores (https://dl.acm.org/doi/10.1145/3620666.3651378)
3. Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages. DOI: https://doi.org/10.1145/2049662.2049663
