```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/spmm <n>
```

can append `-DCMAKE_BUILD_TYPE=Release` to first line if wanted later
