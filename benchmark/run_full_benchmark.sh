#!/usr/bin/env bash
#
# Comprehensive benchmark suite for sptc-dist on 32-node CloudLab cluster.
# Hardware: 32x Intel Xeon D-1548 (8c/16t @ 2.0 GHz), 64 GB RAM, 10 GbE
#
# Runs:
#   1. Dense baseline (single-node, OpenBLAS)
#   2. TiledArray single-node scaling (1, 2, 4, 8 ranks)
#   3. TiledArray multi-node scaling (4, 8, 16, 32 nodes)
#   4. Equation-specific runs (eq0/eq1 with more nodes for large intermediates)
#
# Output: CSV to stdout, diagnostics to stderr.
# Usage: bash run_full_benchmark.sh > results_32node.csv 2>log_32node.txt
set -euo pipefail

PROJ="/proj/perf-model-gpu-PG0/sptc-bench"
BIN_DENSE="${PROJ}/build/benchmark_sptc"
BIN_TA="${PROJ}/build/ta_benchmark"
DATA="${PROJ}/dataset/data_fusedsptc"
HOSTFILE="${PROJ}/hostfile_32"

export OPENBLAS_NUM_THREADS=1
export SPTC_TRIALS=3

MOLECULES=(C2H6 C3H8 C4H10 C5H12 C6H14)

# Helper: run TiledArray benchmark
ta_run() {
    local np="$1" ppn="$2" eq="$3" mol="$4" label="$5"
    local mol_dir="${DATA}/${mol}"
    [[ -d "$mol_dir" ]] || { echo "SKIP $label: $mol_dir not found" >&2; return; }

    echo ">>> TA: $label (np=$np ppn=$ppn eq=$eq mol=$mol)" >&2

    local timeout_s=3600  # 1 hour max per run
    if [[ "$np" -le 8 && "$ppn" -eq "$np" ]]; then
        # Single-node: no hostfile
        SPTC_EQUATIONS="$eq" timeout "$timeout_s" mpirun -np "$np" "$BIN_TA" "$mol_dir" 2>/dev/null || {
            echo "FAILED $label (timeout or error)" >&2
        }
    else
        # Multi-node
        SPTC_EQUATIONS="$eq" timeout "$timeout_s" mpirun -f "$HOSTFILE" -np "$np" -ppn "$ppn" -iface eno1d1 "$BIN_TA" "$mol_dir" 2>/dev/null || {
            echo "FAILED $label (timeout or error)" >&2
        }
    fi
}

# ================================================================
# PART 1: Dense Baseline (single-node, all molecules, all equations)
# ================================================================
# echo "=== Part 1: Dense Baseline ===" >&2
#
# for mol in "${MOLECULES[@]}"; do
#     mol_dir="${DATA}/${mol}"
#     [[ -d "$mol_dir" ]] || continue
#     echo ">>> Dense: $mol" >&2
#     export OPENBLAS_NUM_THREADS=$(nproc)
#     timeout 3600 "$BIN_DENSE" "$mol_dir" 2>/dev/null || {
#         echo "FAILED dense-$mol" >&2
#     }
#     export OPENBLAS_NUM_THREADS=1
# done

# ================================================================
# PART 2: TiledArray Eq2 — scales well, small intermediates
# ================================================================
echo "=== Part 2: TiledArray Eq2 Scaling ===" >&2

# Single-node scaling for C3H8
for np in 1 2 4 8; do
    ta_run $np $np 2 C3H8 "eq2-C3H8-${np}rank-1node"
done

# Multi-node scaling for C3H8 (1 rank per node)
for nnodes in 4 8 16 32; do
    ta_run $nnodes 1 2 C3H8 "eq2-C3H8-${nnodes}rank-${nnodes}node"
done

# Multi-node scaling for C3H8 (2 ranks per node)
for nnodes in 4 8 16 32; do
    np=$((nnodes * 2))
    ta_run $np 2 2 C3H8 "eq2-C3H8-${np}rank-${nnodes}node"
done

# All molecules at best single-node config (1 rank) and multi-node configs
for mol in "${MOLECULES[@]}"; do
    ta_run 1 1 2 "$mol" "eq2-${mol}-1rank"
done

for mol in "${MOLECULES[@]}"; do
    ta_run 4 1 2 "$mol" "eq2-${mol}-4node"
    ta_run 8 1 2 "$mol" "eq2-${mol}-8node"
    ta_run 16 1 2 "$mol" "eq2-${mol}-16node"
    ta_run 32 1 2 "$mol" "eq2-${mol}-32node"
done

# ================================================================
# PART 3: TiledArray Eq0 — large intermediates, benefits from more nodes
# ================================================================
echo "=== Part 3: TiledArray Eq0 ===" >&2

# Eq0 needs multi-node for memory. Start with 4 nodes, scale up.
for nnodes in 4 8 16 32; do
    ta_run $nnodes 1 0 C3H8 "eq0-C3H8-${nnodes}node"
done

# Eq0 with 2 ranks per node
for nnodes in 4 8 16 32; do
    np=$((nnodes * 2))
    ta_run $np 2 0 C3H8 "eq0-C3H8-${np}rank-${nnodes}node"
done

# Larger molecules with more nodes
for mol in C4H10 C5H12 C6H14; do
    ta_run 8 1 0 "$mol" "eq0-${mol}-8node"
    ta_run 16 1 0 "$mol" "eq0-${mol}-16node"
    ta_run 32 1 0 "$mol" "eq0-${mol}-32node"
done

# ================================================================
# PART 4: TiledArray Eq1 — rank-6 intermediate, previously OOM on 4 nodes
# With 32 nodes (2 TB total), may now be feasible
# ================================================================
echo "=== Part 4: TiledArray Eq1 ===" >&2

# Try Eq1 with increasing resources
for nnodes in 8 16 32; do
    ta_run $nnodes 1 1 C3H8 "eq1-C3H8-${nnodes}node"
done

for nnodes in 8 16 32; do
    np=$((nnodes * 2))
    ta_run $np 2 1 C3H8 "eq1-C3H8-${np}rank-${nnodes}node"
done

# If C3H8 works at 32 nodes, try C4H10
ta_run 32 1 1 C4H10 "eq1-C4H10-32node"

echo "=== All benchmarks complete ===" >&2
