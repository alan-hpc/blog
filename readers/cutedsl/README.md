# CuteDSL MHC pre-norm-fn bwd-fused — review bundle

CuteDSL (pure `cutlass` / `cutlass.cute` / `cutlass.utils`, **no quack**) reimplementation of
`tile_kernels.mhc.norm_fn_kernel._mhc_pre_norm_fn_bwd_fused`, developed/validated on the remote
H100 box (10.249.33.13, container `miny_mhc_bench`, cutlass 4.3.4). Targets `n_rms_group == 1`.

## Files
- `harness.py` — standalone correctness (vs inline torch ref) + bench. No tilelang dependency.
- `bench_tilelang.py` — runs the original TileLang kernel for the baseline number.
- `mhc_bwd_fused_cute_v2_vectorized.py` — correct, simple: per-hidden-tile CTA loops token tiles;
  scalar smem GEMMs; 128-bit vectorized x / x_grad copies.
- `mhc_bwd_fused_cute_wgmma_v4.py` — correct: same structure but both GEMMs on bf16 wgmma tensor
  cores (via `cutlass.utils.hopper_helpers`). fn_grad uses hi+lo (2×bf16 ≈ tf32) for precision.
- `DESIGN_wgmma.md`, `DESIGN_2dgrid.md` — design notes / accumulated wgmma gotchas.

## Perf (bench shape: num_tokens=4096, mhc_mult=4, hidden_size=2048 → M3=24, H=8192, H100)
| version | time | bandwidth | correct |
|---|---|---|---|
| TileLang baseline (`_mhc_pre_norm_fn_bwd_fused`) | 326 µs | 419 GB/s | ✅ |
| CuteDSL scalar smem GEMM | 724 µs | 189 GB/s | ✅ |
| CuteDSL + vectorized IO (v2) | 457 µs | 299 GB/s | ✅ |
| CuteDSL + bf16 wgmma (v4) | 447 µs | 305 GB/s | ✅ |
| CuteDSL 2D-grid v1 | 282 µs | 485 GB/s | ✅ |
| **CuteDSL 2D-grid + smem-cut (`mhc_bwd_fused_cute_2dgrid.py`)** | **168 µs** | **813 GB/s** | ✅ |

Bandwidth floor ≈ 45 µs (134 MB of x read + x_grad write at ~3 TB/s).

**Final winner: 168 µs / 818 GB/s (164±3) — ~1.93× faster than TileLang (326 µs), correct on both
`accumulate_x_grad` cases (independently re-verified).**

ncu-guided optimization loop CONVERGED (stopped after 3 consecutive no-improvement rounds):
- **Round 1 — IMPROVED 282→168 µs**: cut shared memory (BN 256→128, eliminated duplicate plain staging
  buffers, fill swizzled smem directly), raising occupancy off the 6.25% smem-bound floor to 18.75%.
- Round 2 (smem/tiling micro-sweep) — no improvement.
- Round 3 (token-grouping K_TOK sweep, vectorized fills, split-K) — no improvement; proved atomics are
  NOT the bottleneck.
- Round 4 (eliminate x-transpose) — no improvement; rigorously proved the transpose does essential
  swizzle-coordinate reconciliation (a wgmma readback test showed direct-load scrambles the token index).

**Root cause of the 168 µs floor (consistent across all rounds):** `_mul_bwd_kernel` is **L1/TEX
instruction-issue bound** (L1/TEX ~87%, DRAM/Compute/Mem-pipes all ~21%), driven by the algorithmically
required scalar operand-staging fills (the x→B-operand transpose + the omg hi/lo bf16 split) needed to
feed the skinny M3=24 contraction GEMM through wgmma's swizzled-smem operand format. Occupancy is
hard-capped at 3 blocks/SM by registers (162) AND smem (61.7 KB) simultaneously. This is an architectural
floor of the wgmma approach for this problem shape, not a tuning miss. The ~45 µs HBM floor is
unreachable here because the work is on-chip-instruction-bound, not HBM-bound (DRAM only ~21%).

`mhc_bwd_fused_cute_2dgrid.py` in this folder is the final 168 µs kernel.

## Architecture of the winning 2D-grid version
- **Kernel A `_norm_bwd`** (tiny): 1D grid over tokens, one token/thread. Precomputes `omg_ws (nt,M3) f32`
  and `sqg_ws (nt,) f32` ONCE into workspace tensors (allocated by the wrapper).
- **Kernel B `_mul_bwd`**: 2D grid `(nt//BM, H//BN)`, **no serial token loop**. Each CTA does one
  (token tile × hidden tile): loads x (vectorized) + omg (split bf16 hi/lo) + fn into swizzled smem,
  runs `G_xg = omg_hi@fn` (+ epilogue `2*x*sqg (+init)`) and stores x_grad, runs
  `G_fng = omg_hi^T@x + omg_lo^T@x` and reduces across token tiles into `fn_grad` via **f32 `nvvm.atomicrmw` FADD**.
  Wrapper `fn_grad.zero_()` before kernel B. Tiles: BM=64, BN=256 (wgmma N≤256), NT=128.

This removed the v4 serial loop + per-iteration swizzle re-staging and the redundant out_grad/out_mul
re-reads, raising occupancy and cutting effective traffic — beating TileLang.

## Run (on the remote box)
```
Y=/data1/yimeng/repo/TensorRT-LLM/.venv-3.12/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages
cd /data1/min.yang/mhc_cute && PYTHONPATH=$Y python harness.py --bench   # cute correctness + bench
cd /data1/min.yang/mhc_cute && PYTHONPATH=$Y python bench_tilelang.py     # TileLang baseline
```
Note: the container's own cutlass 4.5.1 is broken; the `$Y` PYTHONPATH (yimeng's working 4.3.4) is required.
