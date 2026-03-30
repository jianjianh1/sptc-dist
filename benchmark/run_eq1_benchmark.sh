#!/usr/bin/env bash
#
# Eq1 benchmark suite — systematic exploration across molecules and node counts.
# Uses local storage and internal 10.10.1.x network.
#
# Usage: bash run_eq1_benchmark.sh > results_eq1.csv 2>log_eq1.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/build/ta_benchmark"
DATA="${SCRIPT_DIR}/../dataset/data_fusedsptc"
HOSTFILE="${SCRIPT_DIR}/hostfile_32"

export OPENBLAS_NUM_THREADS=1
export SPTC_TRIALS=3
export SPTC_EQUATIONS=1

MOLECULES=(C2H6 C3H8 C4H10 C5H12 C6H14)

ta_run() {
    local np="$1" ppn="$2" mol="$3" label="$4"
    local mol_dir="${DATA}/${mol}"
    [[ -d "$mol_dir" ]] || { echo "SKIP $label: $mol_dir not found" >&2; return; }

    echo ">>> $label (np=$np ppn=$ppn mol=$mol)" >&2

    local timeout_s=3600
    if [[ "$np" -le 8 && "$ppn" -eq "$np" ]]; then
        timeout "$timeout_s" mpirun -np "$np" "$BIN" "$mol_dir" 2>/dev/null || {
            echo "FAILED $label (timeout or error)" >&2
        }
    else
        timeout "$timeout_s" mpirun -f "$HOSTFILE" -np "$np" -ppn "$ppn" -iface eno1d1 "$BIN" "$mol_dir" 2>/dev/null || {
            echo "FAILED $label (timeout or error)" >&2
        }
    fi
}

# ================================================================
# C2H6 — smallest molecule, known to work at 32 nodes
# ================================================================
echo "=== C2H6 ===" >&2
ta_run 1 1 C2H6 "eq1-C2H6-1rank"
for nnodes in 4 8 16 32; do
    ta_run $nnodes 1 C2H6 "eq1-C2H6-${nnodes}node"
done

# ================================================================
# C3H8 — previously OOM at all configs, try with more nodes
# ================================================================
echo "=== C3H8 ===" >&2
for nnodes in 4 8 16 32; do
    ta_run $nnodes 1 C3H8 "eq1-C3H8-${nnodes}node"
done

# ================================================================
# C4H10 — try larger node counts
# ================================================================
echo "=== C4H10 ===" >&2
for nnodes in 8 16 32; do
    ta_run $nnodes 1 C4H10 "eq1-C4H10-${nnodes}node"
done

# ================================================================
# C5H12 — try at 16 and 32 nodes
# ================================================================
echo "=== C5H12 ===" >&2
for nnodes in 16 32; do
    ta_run $nnodes 1 C5H12 "eq1-C5H12-${nnodes}node"
done

# ================================================================
# C6H14 — try at 32 nodes only
# ================================================================
echo "=== C6H14 ===" >&2
ta_run 32 1 C6H14 "eq1-C6H14-32node"

echo "=== All eq1 benchmarks complete ===" >&2
