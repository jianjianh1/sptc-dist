# TiledArray Limitations for Distributed Sparse Tensor Contraction

This document provides a fair, detailed assessment of TiledArray's limitations when used for sparse tensor contractions in the DLPNO-CCSD context. All findings are based on our experimental benchmarks (C3H8 molecule, three contraction equations), TiledArray source code analysis (v1.1.0), and published GitHub issues.

---

## 1. Tile-Level-Only Sparsity

TiledArray handles sparsity **exclusively at the tile (block) level**. The `SparseShape<float>` class stores a single Frobenius norm per tile. If a tile's per-element norm falls below a threshold, the entire tile is treated as zero and skipped. Within a non-zero tile, all elements are stored densely -- there is no compressed format (CSR, COO, etc.) inside tiles.

**Maintainer confirmation (GitHub issue [#496](https://github.com/ValeevGroup/tiledarray/issues/496)):** Eduard Valeyev stated: *"In TA we focus on the most common type of structured sparsity pattern (block-sparse tensors) that still allows to attain high performance due to tensor blocks being themselves dense, hence resulting in high arithmetic intensity kernels in most important cases."* He explicitly stated TiledArray is not useful for unstructured (element-level) sparsity, recommending TACO or CoNST instead.

**Impact on DLPNO-CCSD:** DLPNO's pair-specific domains create irregular, element-level sparsity patterns that don't align well with uniform tiling. In our benchmarks, the g0 tensor (the largest, growing quadratically) has **0% tile-level sparsity** -- every tile contains at least some non-zero elements, so TiledArray stores it fully dense. The c1 and c2 tensors achieve 72-74% tile-level sparsity, but the remaining non-zero tiles are stored as full dense blocks even if internally sparse.

**Source:** `src/TiledArray/sparse_shape.h`

---

## 2. No Intermediate Fusion / Streaming

TiledArray has **no mechanism to fuse consecutive contractions** or stream tiles through a pipeline. Each binary contraction via `einsum()` or `operator*` produces a fully materialized `DistArray`. There is no cross-expression lazy evaluation.

This is the fundamental cause of out-of-memory failures:

| Equation | Intermediate | Shape (C3H8) | Dense Size | Tile Sparsity | Actual Memory |
|----------|-------------|--------------|------------|---------------|---------------|
| **Eq0 I1** | `(k, i, a, j, b)` | 483 x 13 x 898 x 13 x 898 | 520 GB | 91.5% | ~26 GB/node on 4 nodes |
| **Eq1 I1** | `(j, k, i, a, p, b)` | rank-6 | >> 520 GB | — | OOM on 4 x 64 GB |

Our dense CBLAS baseline avoids this by **fusing stages 2+3** into a loop over small occupied indices (`ni` ~ 13), performing GEMM on slices and accumulating into the result without ever materializing the full intermediate. TiledArray provides no equivalent fusion mechanism.

The `TA::foreach` API (`src/TiledArray/conversions/foreach.h`) requires all inputs to share the same `TiledRange`, making it unsuitable for fusing operations on tensors with different dimensionalities. The `make_array` API could theoretically enable lazy tile computation, but calling `array.find(idx).get()` inside its lambda blocks a MADNESS worker thread, risking task pool deadlock.

---

## 3. Einsum is Binary Only

The `einsum()` function takes exactly **two tensor operands**:

```cpp
auto einsum(TsrExpr<ArrayA_> A, TsrExpr<ArrayB_> B,
            std::tuple<Index<std::string>, Indices...> result_indices, World& world);
```

There is no 3-operand or N-operand einsum. A ternary `dot(A, B, C)` function exists but only produces a **scalar** result. Multi-tensor contractions (like our 4-tensor DLPNO-CCSD equations) must be manually decomposed into binary stages with explicit intermediate materialization.

TiledArray provides **no automatic contraction order optimization**. The user must choose the intermediate ordering and tiling, which directly affects memory consumption and performance.

**Source:** `src/TiledArray/einsum/tiledarray.h`, line ~411

---

## 4. Mixed Hadamard + Contraction Bug

TiledArray's `operator*` expression engine **crashes** (SIGFPE or SIGSEGV in `SparseShape::perm`) when an index appears in both operands as a non-contracted (Hadamard) index alongside other indices that are contracted. The source code explicitly documents this limitation:

```cpp
// TODO support general products that involve fused, contracted, and free indices
// N.B. Currently only 2 types of products are supported:
// - Hadamard product (in which all indices are fused), and,
// - pure contraction (>=1 contracted, 0 fused, >=1 free indices)
```

**Workaround:** Use `einsum()` which accepts explicit result indices and handles mixed patterns correctly.

**Affected stages in our benchmark:** Eq2 stages 2-3, Eq0 stage 3, Eq1 stages 2-3 -- all require the `einsum()` workaround.

**Source:** `src/TiledArray/expressions/mult_engine.h`, lines 286-291

---

## 5. Einsum Hadamard Loop: MPI Communicator Split Per Tile

When `einsum()` encounters Hadamard (shared non-contracted) indices, it uses a tile-slicing loop that calls **`MPI_Comm_Split` for each Hadamard tile index** to create sub-communicators. For each sub-world, it extracts and reshapes tiles into temporary sub-arrays, performs a binary SUMMA contraction, collects results, and fences.

For H Hadamard tiles, this means H collective communicator splits and H separate contractions. This is correct but adds significant overhead, especially when the number of Hadamard tiles is large.

**Source:** `src/TiledArray/einsum/tiledarray.h`, lines 795-854

---

## 6. Communication and Load Balancing

### Static Tile Distribution

Both dense and sparse policies use `BlockedPmap` as the default process map. Tiles are assigned to processes in contiguous ordinal blocks of approximately N/P tiles. This static assignment does not consider tile sparsity -- zero tiles are "owned" by a process even though they have no data, leading to potential load imbalance when sparsity is unevenly distributed.

**Source:** `src/TiledArray/policies/sparse_policy.h`, line 43

### Input Tiles Always Evaluated

GitHub issue [#80](https://github.com/ValeevGroup/tiledarray/issues/80) (open): *"All input tiles appear to always be evaluated and communicated regardless of output shape."* Even when the output shape indicates certain tiles are zero (and thus don't need to be computed), the SUMMA evaluator eagerly evaluates and broadcasts input tiles. This causes quadratic scaling for problems where the output is much sparser than the input.

### Memory Leakage in SUMMA

GitHub issue [#86](https://github.com/ValeevGroup/tiledarray/issues/86) (open): Eduard Valeyev documented *"uncontrolled memory use in SUMMA."* With 12 threads, ~800 MB of virtual memory was not released after contraction, with leakage scaling linearly with thread count. This remains unfixed.

### Our Multi-Rank Scaling Results (C3H8, Eq2)

| Config | Total Time | Relative to 1 Rank |
|--------|-----------|-------------------|
| 1 rank, 1 node | 1.18s | 1.0x |
| 2 ranks, 1 node | 6.36s | 5.4x slower |
| 4 ranks, 1 node | 8.18s | 6.9x slower |
| 4 ranks, 4 nodes | 2.33s | 2.0x slower |

Multi-rank execution is **slower** than single-rank for this problem size. MPI communication overhead (tile broadcasts, reductions, communicator management) overwhelms the ~1s computation time. Multi-node helps only for memory distribution (Eq0 requires 4 nodes to fit I1).

---

## 7. Build and Deployment Issues

### MPICH LTO Deadlock (Undocumented)

Ubuntu 22.04's MPICH 4.0 package wraps `mpicxx` with `-flto=auto`. GCC 11's Link-Time Optimization corrupts `inline static thread_local` variables in MADNESS's `DQueue` task prebuffer (`dqueue.h`), causing all Einstein-notation expressions to deadlock after `fill()` + `fence()`. Even trivial operations like `B("i,j") = 2.0 * A("i,j")` hang indefinitely.

**Fix:** Add `-fno-lto` to all CMake builds. This is **not documented** in TiledArray's official installation guide.

### OpenBLAS Thread Oversubscription (Undocumented)

TiledArray only throttles BLAS threading for Intel MKL (via `set_num_threads(1)` in `tiledarray.cpp`, line 112). With OpenBLAS, each MADNESS worker thread spawns a full OpenBLAS thread pool, causing severe oversubscription. Users must manually call `openblas_set_num_threads(1)` before TiledArray initialization.

### TBB Backend Incompatibility

The TBB task backend (`-DMADNESS_TASK_BACKEND=TBB`) is incompatible with oneAPI TBB (2021+). TiledArray uses legacy TBB headers (`tbb/tbb_stddef.h`) and API that was removed in the oneAPI transition. The Pthreads backend works correctly with the `-fno-lto` fix.

### Single-Character Index Names

TiledArray's Einstein notation parser only supports **single-character index names**. Multi-character names like `"i2,m1,k1"` cause crashes. This limits expressiveness when mapping from equations with named subscripts.

---

## 8. SparseShape Threshold Sensitivity

The sparse threshold is a global mutable static variable with lossy, irreversible behavior:

> *"If tile's scaled norm is below threshold, its scaled norm is set to zero and thus lost forever. E.g. `shape.scale(1e-10).scale(1e10)` does not in general equal `shape`."*
> -- `sparse_shape.h`, lines 63-65

The default threshold (0.0) combined with per-element Frobenius norm scaling can incorrectly prune tiles. Large tiles with small per-element norms are more aggressively pruned than small tiles. Our benchmark sets `threshold(1e-10f)` to prevent data loss, but this means no pruning of near-zero tiles, reducing sparsity benefits.

---

## 9. Known Open GitHub Issues

| Issue | Description |
|-------|-------------|
| [#80](https://github.com/ValeevGroup/tiledarray/issues/80) | All input tiles evaluated/communicated regardless of output shape |
| [#86](https://github.com/ValeevGroup/tiledarray/issues/86) | Uncontrolled memory use in SUMMA (leakage scales with threads) |
| [#184](https://github.com/ValeevGroup/tiledarray/issues/184) | Reduction with sparse tensors gives incorrect results for zero tiles |
| [#282](https://github.com/ValeevGroup/tiledarray/issues/282) | operator* for Tensor-of-Tensors incomplete |
| [#496](https://github.com/ValeevGroup/tiledarray/issues/496) | Maintainer confirms TA not suitable for element-level sparsity |

---

## 10. Where TiledArray Excels

For a fair comparison, TiledArray's strengths should be acknowledged:

- **Productivity:** Einstein notation (`C("i,j") = A("i,k") * B("k,j")`) is genuinely expressive and productive for rapid prototyping.
- **Block-sparse GEMM throughput:** Dense tiles hit peak BLAS performance; the tile-level sparsity skip avoids unnecessary computation.
- **Distributed memory:** Automatic tile distribution and SUMMA-based contraction scale to many nodes for large, block-sparse problems.
- **Tile-level sparsity gains can be dramatic:** Our Eq2 achieved a 22x speedup over the dense baseline due to 72-92% tile-level sparsity.

---

## 11. Summary: Key Limitations for DLPNO-CCSD Workloads

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| Tile-level-only sparsity | Cannot exploit intra-tile zeros; g0 tensor stored fully dense despite element-level sparsity | None in TiledArray; element-level sparse library needed |
| No intermediate fusion | Eq0/Eq1 OOM due to large materialized intermediates | Multi-node distribution (Eq0) or manual fusion outside TA (not supported) |
| Binary-only einsum | 4-tensor equations require 3 staged contractions with manual ordering | Manual decomposition |
| Hadamard+contraction crash | operator* unsupported for mixed patterns | Use einsum() with explicit result indices |
| No contraction order optimization | User must choose intermediate ordering affecting memory and performance | Manual analysis |
| Static load balancing | No sparsity-aware tile distribution | None |
| Build issues (-flto, OpenBLAS) | Undocumented deadlocks on common platforms | Manual flags: `-fno-lto`, `openblas_set_num_threads(1)` |
| MPI overhead for small problems | Multi-rank slower than single-rank on small problems | Use single rank for small problems |

---

## References

- [TiledArray GitHub](https://github.com/ValeevGroup/tiledarray)
- [Issue #80: Input tiles always evaluated](https://github.com/ValeevGroup/tiledarray/issues/80)
- [Issue #86: Uncontrolled SUMMA memory](https://github.com/ValeevGroup/tiledarray/issues/86)
- [Issue #496: Not suitable for element-level sparsity](https://github.com/ValeevGroup/tiledarray/issues/496)
- [Issue #514: Einsum Hadamard permutation bug](https://github.com/ValeevGroup/tiledarray/issues/514)
- [Calvin et al., "Scalable Task-Based Algorithm for Block-Rank-Sparse Matrix Multiplication" (2015)](https://dl.acm.org/doi/10.1145/2833179.2833186)
- [Peng et al., "Massively Parallel CCSD Using TiledArray" (2016)](https://pubs.acs.org/doi/10.1021/acs.jpca.6b10150)
- [Mutlu et al., "Minimum Cost Loop Nests for Sparse Tensor Network Contraction" (CoNST, 2024)](https://arxiv.org/abs/2307.05740)
