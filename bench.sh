#!/usr/bin/env bash

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/

for ((i=2; i<=16384; i*=2))
do
    echo "[i = $i]"
    ./build/spmm $i
done
