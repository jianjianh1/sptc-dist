#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DATA_DIR="${SCRIPT_DIR}/../dataset/data_fusedsptc"

# Build
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "$SCRIPT_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"

# Use all cores for BLAS
export OPENBLAS_NUM_THREADS=$(nproc)

# Run for each molecule, smallest first
MOLECULES=(C3H8 C4H10 C5H12 C6H14)

ARGS=()
for mol in "${MOLECULES[@]}"; do
    if [ -d "$DATA_DIR/$mol" ]; then
        ARGS+=("$DATA_DIR/$mol")
    else
        echo "WARNING: $DATA_DIR/$mol not found, skipping" >&2
    fi
done

if [ ${#ARGS[@]} -eq 0 ]; then
    echo "ERROR: no molecule data directories found" >&2
    exit 1
fi

echo "Running benchmark for: ${MOLECULES[*]}" >&2
"$BUILD_DIR/benchmark_sptc" "${ARGS[@]}"
