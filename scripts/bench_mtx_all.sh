#!/usr/bin/env bash
set -euo pipefail

NCOLS="${1:-128}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/ -j

for mtx in data/*/*.mtx; do
    name=$(basename "$mtx" .mtx)
    echo "=== $name (B ncols=$NCOLS) ==="
    ./build/spmm "$mtx" "$NCOLS"
    echo ""
done
