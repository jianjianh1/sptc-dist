#!/usr/bin/env bash
#
# Scaling study for TiledArray distributed SPTC benchmark.
# Runs strong scaling (fixed molecule, vary ranks) and weak scaling
# (larger molecules with more ranks).
#
# Usage:
#   bash run_scaling_study.sh [strong|weak|all]
#
# Output: CSV to stdout, diagnostics to stderr.
# Redirect: bash run_scaling_study.sh all > scaling_results.csv 2>scaling_log.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${SCRIPT_DIR}/build/ta_benchmark"
DATA_DIR="${SCRIPT_DIR}/../dataset/data_fusedsptc"

export OPENBLAS_NUM_THREADS=1
export CMAKE_EXTRA_FLAGS="-DCMAKE_CXX_FLAGS=-fno-lto -DCMAKE_C_FLAGS=-fno-lto"
export SPTC_TRIALS="${SPTC_TRIALS:-3}"

MODE="${1:-all}"

# Build if needed
if [[ ! -x "$BENCHMARK" ]]; then
    echo "Building ta_benchmark..." >&2
    cmake -B "${SCRIPT_DIR}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-fno-lto" \
        -DCMAKE_C_FLAGS="-fno-lto" \
        "${SCRIPT_DIR}" >&2
    cmake --build "${SCRIPT_DIR}/build" -j"$(nproc)" --target ta_benchmark >&2
fi

run_benchmark() {
    local np="$1"
    local molecule="$2"
    local mol_dir="${DATA_DIR}/${molecule}"

    if [[ ! -d "$mol_dir" ]]; then
        echo "SKIP: ${mol_dir} not found" >&2
        return
    fi

    echo ">>> Running np=${np} molecule=${molecule}" >&2
    mpirun -np "$np" "$BENCHMARK" "$mol_dir"
}

# ─── Strong Scaling: fixed C3H8, vary ranks ────────────────────────────────
strong_scaling() {
    echo "=== Strong Scaling (C3H8) ===" >&2
    for NP in 1 2 4 8 16; do
        run_benchmark "$NP" "C3H8"
    done
}

# ─── Weak Scaling: scale molecule with ranks ────────────────────────────────
weak_scaling() {
    echo "=== Weak Scaling ===" >&2
    declare -a MOLECULES=(C2H6 C3H8 C4H10 C5H12 C6H14)
    declare -a RANKS=(1 2 4 8 16)

    for i in "${!MOLECULES[@]}"; do
        run_benchmark "${RANKS[$i]}" "${MOLECULES[$i]}"
    done
}

# ─── Main ───────────────────────────────────────────────────────────────────
case "$MODE" in
    strong)
        strong_scaling
        ;;
    weak)
        weak_scaling
        ;;
    all)
        strong_scaling
        weak_scaling
        ;;
    *)
        echo "Usage: $0 [strong|weak|all]" >&2
        exit 1
        ;;
esac

echo "Done." >&2
