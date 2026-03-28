#!/usr/bin/env bash
#
# Run all feasible TiledArray benchmarks and record results.
# Output: CSV to stdout, diagnostics to stderr.
#
# Usage: bash run_all_benchmarks.sh > results.csv 2>log.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/build/ta_benchmark"
DATA="${SCRIPT_DIR}/../dataset/data_fusedsptc"
HOSTFILE="${SCRIPT_DIR}/hostfile"

export OPENBLAS_NUM_THREADS=1
export SPTC_TRIALS=3

run() {
    local np="$1" ppn="$2" eq="$3" mol="$4" label="$5"
    local mol_dir="${DATA}/${mol}"
    [[ -d "$mol_dir" ]] || { echo "SKIP $label: $mol_dir not found" >&2; return; }

    echo ">>> $label (np=$np ppn=$ppn eq=$eq mol=$mol)" >&2

    if [[ "$np" -le 8 && "$ppn" -eq "$np" ]]; then
        # Single-node: no hostfile needed
        SPTC_EQUATIONS="$eq" timeout 1800 mpirun -np "$np" "$BIN" "$mol_dir" 2>/dev/null || {
            echo "FAILED $label (timeout or error)" >&2
        }
    else
        # Multi-node
        SPTC_EQUATIONS="$eq" timeout 1800 mpirun -f "$HOSTFILE" -np "$np" -ppn "$ppn" -iface eno1d1 "$BIN" "$mol_dir" 2>/dev/null || {
            echo "FAILED $label (timeout or error)" >&2
        }
    fi
}

# ──────────────────────────────────────────────────────────
# Eq2: small intermediates, works everywhere
# ──────────────────────────────────────────────────────────

# Single-node scaling (C3H8)
run 1 1 2 C3H8 "eq2-C3H8-1rank"

# Multi-node (C3H8)
run 4 1 2 C3H8 "eq2-C3H8-4node"

# Larger molecules, 1 rank
run 1 1 2 C4H10 "eq2-C4H10-1rank"
run 1 1 2 C5H12 "eq2-C5H12-1rank"
run 1 1 2 C6H14 "eq2-C6H14-1rank"

# Larger molecules, 4 nodes
run 4 1 2 C4H10 "eq2-C4H10-4node"
run 4 1 2 C5H12 "eq2-C5H12-4node"
run 4 1 2 C6H14 "eq2-C6H14-4node"

# ──────────────────────────────────────────────────────────
# Eq0: large intermediate, needs multi-node for C3H8+
# ──────────────────────────────────────────────────────────

# Multi-node (C3H8)
run 4 1 0 C3H8 "eq0-C3H8-4node"
run 8 2 0 C3H8 "eq0-C3H8-8rank-4node"

# Larger molecules, 4 nodes
run 4 1 0 C4H10 "eq0-C4H10-4node"

# ──────────────────────────────────────────────────────────
# Eq1: rank-6 intermediate, OOMs on 4 nodes for C3H8
# Try with 8 ranks (2 per node) to distribute more
# ──────────────────────────────────────────────────────────

run 8 2 1 C3H8 "eq1-C3H8-8rank-4node"
run 16 4 1 C3H8 "eq1-C3H8-16rank-4node"

echo "=== All benchmarks complete ===" >&2
