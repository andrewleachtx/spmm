```sh
cmake -S . -B build
cmake --build build -j
./build/spmm
```

can append `-DCMAKE_BUILD_TYPE=Release` to first line if wanted later
