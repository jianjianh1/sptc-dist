# fastcc_dlpno Context Dump

## Scope
This file captures the current working understanding of:
- eq0 / eq1 / eq2 operator definitions and implementation variants
- dataset layout in `data_fusedsptc`

---

## Eq0
Canonical comment form:
`R(i1, i2, a1, a2) = g0(u1,u2,k1) * c(i1,u1,a1) * c(i2,u2,a2) * g1(i2,i1,k1)` (summed over `u1,u2,k1`)

Common staged view:
1. `I0(u2,k1,i1,a1) = g0(u1,u2,k1) * c(i1,u1,a1)` (contract `u1`)
2. `I1(k1,i1,a1,i2,a2) = I0(u2,k1,i1,a1) * c(i2,u2,a2)` (contract `u2`)
3. `R(i1,i2,a1,a2) = g1(i2,i1,k1) * I1(k1,i1,a1,i2,a2)` (contract `k1`)

Key variants:
- `run_eq0_constfused`
- `run_eq0_constblas`
- `run_eq0_constblas_mklspgemm`
- `run_eq0_constblas_densedense`
- `run_eq0_unfused`
- `run_eq0_unfused_tileindexed`

## Eq1
Canonical comment form:
`R(i1, i3, i2, a2, a3) = g0(u1,u2,k1) * c(i2,i1,u1,a2) * c(i1,i3,u2,a3) * g1(i2,i3,k1)` (summed over `u1,u2,k1`)

Staged view:
1. `I0(u2,k1,i2,i1,a2) = g0(u1,u2,k1) * c(i2,i1,u1,a2)` (contract `u1`)
2. `I1(i1,k1,i2,a2,i3,a3) = I0(u2,k1,i2,i1,a2) * c(i1,i3,u2,a3)` (contract `u2`)
3. `R(i2,i3,i1,a2,a3) = I1(i1,k1,i2,a2,i3,a3) * g1(i2,i3,k1)` (contract `k1`)

Key variants:
- `run_eq1_constfused`
- `run_eq1_constblas`
- `run_eq1_constblas_parallel`
- `run_eq1_constblas_densedense`
- `run_eq1_fused`
- `run_eq1_1dfused`
- `run_eq1_unfused`

## Eq2
Canonical comment form:
`R(i3,i2,a2,i5,a3) = g(i2,m1,k1) * g(i3,m2,k1) * c1(i2,m1,a2) * c2(i3,i5,m2,a3)` (summed over `k1,m1,m2`)

Staged view:
1. `I0(i2,m1,i3,m2) = g(i2,m1,k1) * g(i3,m2,k1)` (contract `k1`)
2. `I1(i2,i3,m2,a2) = I0(i2,m1,i3,m2) * c1(i2,m1,a2)` (contract `m1`)
3. `R(i3,i2,a2,i5,a3) = I1(i2,i3,m2,a2) * c2(i3,i5,m2,a3)` (contract `m2`)

Key variants:
- `run_eq2_constfused`
- `run_eq2_constfused_blas`
- `run_eq2_constblas_densedense`
- `run_eq2_constfused_mklspgemm`
- `run_eq2_unfused`

---

## 4) What `main` currently executes in `benchmark_ed.cc`

`main -> run_all_molecules(molecule)` currently enables:
- `run_eq1_constfused(g0, g1, c2)`
- `run_eq1_constblas(g0, g1, c2)`

Most eq0/eq2 calls inside `run_all_molecules` are present but commented out.

---

## 5) Dataset Layout (`data_fusedsptc`)

Directory is a symlink in repo root:
- `data_fusedsptc -> /home/jianjian/data/data_fusedsptc`

Molecule folders observed:
- `C2H6`, `C3H8`, `C4H10`, `C5H12`, `C6H14`, `C7H16`, `C8H18`, `C9H20`

Common tensor files per molecule:
- `g_m_1_m_2_Κ_1.txt`  (used as `g0` in eq0/eq1)
- `g_i_1_i_2_Κ_1.txt`  (used as `g1` in eq0/eq1)
- `g_i_1_m_1_Κ_1.txt`  (used as `g2`/`g` in eq2)
- `C_m_1_a_1_i_1.txt`  (used as `c1` in eq0/eq2)
- `C_m_1_a_1_i_1_i_2.txt` (used as `c2` in eq1/eq2)

Additional per-molecule artifacts may exist:
- `input.json`
- `<molecule>.out`
- `script.sh`
- some hidden metadata files (`._*`, `.txt`, etc.)

---

## 6) Tensor Size Inventory (computed from file scan)

Each line shows: `dims`, `nnz`, inferred `shape` (max index + 1 per axis).

### C2H6
- `g_m_1_m_2_Κ_1.txt`: dims=3, nnz=7,091,712, shape=144,144,342
- `g_i_1_i_2_Κ_1.txt`: dims=3, nnz=16,758, shape=9,9,342
- `g_i_1_m_1_Κ_1.txt`: dims=3, nnz=344,736, shape=9,144,342
- `C_m_1_a_1_i_1.txt`: dims=3, nnz=85,284, shape=9,144,609
- `C_m_1_a_1_i_1_i_2.txt`: dims=4, nnz=383,016, shape=9,9,144,585

### C3H8
- `g_m_1_m_2_Κ_1.txt`: dims=3, nnz=19,708,332, shape=202,202,483
- `g_i_1_i_2_Κ_1.txt`: dims=3, nnz=48,300, shape=13,13,483
- `g_i_1_m_1_Κ_1.txt`: dims=3, nnz=975,660, shape=13,202,483
- `C_m_1_a_1_i_1.txt`: dims=3, nnz=159,352, shape=13,202,898
- `C_m_1_a_1_i_1_i_2.txt`: dims=4, nnz=938,488, shape=13,13,202,876

### C4H10
- `g_m_1_m_2_Κ_1.txt`: dims=3, nnz=42,182,400, shape=260,260,624
- `g_i_1_i_2_Κ_1.txt`: dims=3, nnz=105,456, shape=17,17,624
- `g_i_1_m_1_Κ_1.txt`: dims=3, nnz=2,109,120, shape=17,260,624
- `C_m_1_a_1_i_1.txt`: dims=3, nnz=239,128, shape=17,260,1295
- `C_m_1_a_1_i_1_i_2.txt`: dims=4, nnz=1,579,972, shape=17,17,260,1262

### C5H12
- `g_m_1_m_2_Κ_1.txt`: dims=3, nnz=77,359,860, shape=318,318,765
- `g_i_1_i_2_Κ_1.txt`: dims=3, nnz=195,840, shape=21,21,765
- `g_i_1_m_1_Κ_1.txt`: dims=3, nnz=3,892,320, shape=21,318,765
- `C_m_1_a_1_i_1.txt`: dims=3, nnz=313,486, shape=21,318,1688
- `C_m_1_a_1_i_1_i_2.txt`: dims=4, nnz=2,234,706, shape=21,21,318,1655

### C6H14
- `g_m_1_m_2_Κ_1.txt`: dims=3, nnz=127,481,856, shape=376,376,906
- `g_i_1_i_2_Κ_1.txt`: dims=3, nnz=327,066, shape=25,25,906
- `g_i_1_m_1_Κ_1.txt`: dims=3, nnz=6,472,464, shape=25,376,906
- `C_m_1_a_1_i_1.txt`: dims=3, nnz=385,430, shape=25,376,1839
- `C_m_1_a_1_i_1_i_2.txt`: dims=4, nnz=2,882,690, shape=25,25,376,1806

### C7H16
- `g_m_1_m_2_Κ_1.txt`: dims=3, nnz=188,176,164, shape=434,434,1047
- `g_i_1_i_2_Κ_1.txt`: dims=3, nnz=506,748, shape=29,29,1047
- `g_i_1_m_1_Κ_1.txt`: dims=3, nnz=9,925,536, shape=29,434,1047
- `C_m_1_a_1_i_1.txt`: dims=3, nnz=455,264, shape=29,434,2272
- `C_m_1_a_1_i_1_i_2.txt`: dims=4, nnz=3,517,490, shape=29,29,434,2240

### C8H18
- `g_m_1_m_2_Κ_1.txt`: dims=3, nnz=256,583,832, shape=492,492,1188
- `g_i_1_i_2_Κ_1.txt`: dims=3, nnz=737,220, shape=33,33,1188
- `g_i_1_m_1_Κ_1.txt`: dims=3, nnz=14,058,564, shape=33,492,1188
- `C_m_1_a_1_i_1.txt`: dims=3, nnz=525,052, shape=33,492,2567
- `C_m_1_a_1_i_1_i_2.txt`: dims=4, nnz=4,156,172, shape=33,33,492,2533

