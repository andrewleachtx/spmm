#!/bin/bash
# Download a matrix from the SuiteSparse Matrix Collection.
# Usage: ./scripts/download_mtx.sh <group/name>
# Example: ./scripts/download_mtx.sh HB/bcsstk01
#          ./scripts/download_mtx.sh SNAP/amazon0312

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <group/name>"
    echo "Example: $0 HB/bcsstk01"
    exit 1
fi

GROUP_NAME="$1"
NAME=$(basename "$GROUP_NAME")
DATA_DIR="$(dirname "$0")/../data"

mkdir -p "$DATA_DIR"

URL="https://suitesparse-collection-website.herokuapp.com/MM/${GROUP_NAME}.tar.gz"

echo "Downloading ${GROUP_NAME} from SuiteSparse..."
TMPFILE=$(mktemp /tmp/spmm_mtx_XXXXXX.tar.gz)
trap "rm -f $TMPFILE" EXIT

curl -fSL "$URL" -o "$TMPFILE"

echo "Extracting to ${DATA_DIR}/${NAME}/ ..."
tar xzf "$TMPFILE" -C "$DATA_DIR"

MTX_FILE="${DATA_DIR}/${NAME}/${NAME}.mtx"
if [ -f "$MTX_FILE" ]; then
    echo "Done: ${MTX_FILE}"
else
    echo "Warning: expected ${MTX_FILE} not found. Contents:"
    ls -R "${DATA_DIR}/${NAME}/"
fi
