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
