# Plan: Hopper (sm90, H100/H200) BF16 backend for dist_moe

## Context

dist_moe (`meta-pytorch/dist_moe`; fork `garrett361/dist_moe`, remotes `origin` = fork, `upstream` = meta-pytorch) only runs on Blackwell (sm100/sm103). Its BF16 expert GEMMs are fused CuTe DSL kernels built on tcgen05 MMA with TMEM accumulators and 2-CTA clusters, none of which exist on Hopper. The goal is BF16 training and inference on H100/H200 with the same public API (`dist_moe.Config`, `dist_moe.create_context`, `dist_moe.routed_experts`), the same registered custom ops, and the same memory planner. FP8 on Hopper is out of scope (possible later work; Hopper has no block-scaled MMA and no FP4 tensor cores, so MXFP8/NVFP4 parity is not achievable).

All line numbers below are at `main` = `caf454c`; re-locate by function name if they drift. All paths are relative to the repo root unless absolute.

Decisions already made with the user:
- Three implementation steps, each on its own stacked `git tree` branch and PR: Step 1 unfused Triton backend on `feat/sm90-bf16-triton` (detailed below; this `PLAN.md` is that branch's first commit), Step 2 fused Triton kernels on `feat/sm90-bf16-triton-fused` (moderate detail), Step 3 CuTe DSL sm90 port on `feat/sm90-bf16-cute` (high level; details filled in from Step 1 and 2 measurements).
- Work runs on an H200 cluster. Never `git push` without the user's explicit approval each time.
- This `PLAN.md` commit is for handoff only. Drop it from the stack before opening upstream PRs: `AGENTS.md` requires public-release-clean content (no private project names, infrastructure, or code names in source, tests, docs, or commit messages).
- Follow `AGENTS.md` conventions: Google-style docstrings on every non-kernel callable, inline comments only for non-obvious ownership/distributed/memory/graph invariants, fixed-shape tensor suffixes (`_TD`, `_EFD`, ...), validation order from `tests/AGENTS.md`, never weaken an assertion. Never write an em-dash into any file.

## What already works on sm90 and what does not (verified by reading `caf454c`)

Portable today (Triton, PyTorch, or CUDA driver code, no capability checks on the BF16 path):
- Routing: `copy_routing_and_dispatch` / `copy_routing` (`dist_moe/kernels/triton/conditional_copy.py:158`, `:133`), `dist_dispatch_routing` (`dist_moe/_routing_metadata.py:718`, kernels in `dist_moe/kernels/triton/dist_dispatch_routing_kernel.py`). Output `DistDispatchRoutingOutput` (`dist_moe/_routing_metadata.py:46`) holds int64 absolute-address pointer tables: `fwd_gather_ptrs[m]` = peer dispatch base + `t * dispatch_stride`; `scatter_ptrs[m]` = peer combine base + `(t*K + k) * D * 2`; `bwd_gather_ptrs[m]` = peer dispatch base + `(t*K + k) * stride`. Pointer value 0 is a padding sentinel (gathers zero-fill, scatters skip). Rows are expert-sorted; `num_tokens_per_local_experts` int32[E_local]; receive count is `num_tokens_per_rank[local_rank]`, device-only. Entries past the routed rows are undefined.
- Symmetric memory (`dist_moe/_buffers.py:228-264`, `SymmetricMemoryBuffer.create` `:429`, `_CommunicationBuffers.create` `:574`): routing int16, dispatch bf16 `[Tmax, K, D]`, combine bf16 `[Tmax, K, D]`. `buffer_ptrs_tensor` / `signal_pad_ptrs_tensor` int64 device tensors indexed by rank.
- EP barrier `_ep_barrier` (`dist_moe/api.py:1814`) -> `symmetric_memory_barrier` (`dist_moe/_triton_ops.py:62`): NVLS `multimem.red.release.sys` with automatic fallback to unicast signal pads (`dist_moe/kernels/triton/barrier.py`). Valid on sm90.
- Triton `swiglu_fwd` / `swiglu_bwd` (`dist_moe/kernels/triton/swiglu.py:272`, `:671`), `scale_and_sum`, `broadcast_and_scale`, `reduce_from_topk` (`dist_moe/kernels/triton/broadcast_n_reduction.py`), activation copies (`copy_dispatch_to_activation` etc., `conditional_copy.py:238`), device planner kernels (`dist_moe/_activation_buffer_planner_kernel.py`), VMM (`dist_moe/_vmm.py`, CUDA driver only).
- Fused RMSNorm combine (`dist_moe/kernels/norm/fused_rmsnorm_combine.py`) is CuTe but already has sm90 branches (`:375-397`); its parallel-TOPK schedules are Blackwell-only (`:92`).

Blackwell-only on the BF16 path: exactly six CuTe launchers, called from `_Bf16Autograd.forward` (`dist_moe/api.py:1845`) and `_Bf16Autograd._backward_impl` (`dist_moe/api.py:2205`):

| ID | Call site | Launcher (`dist_moe/kernels/...`) | Computes | Fused communication |
|---|---|---|---|---|
| F6, B4 | `api.py:2076`, `:2358` | `dist_grouped_gemm_fprop_dispatch` (`dist_grouped_gemm.py:1148`) | gather x rows from peers into `x_gathered` [M,D], h1 = x_gathered @ W13[e].T [M,2F], optional SwiGLU h2 | gather via `fwd_gather_ptrs` in producer |
| F8, B6 | `api.py:2111`, `:2387` | `dist_grouped_gemm_fprop_combine` (`:1498`) | h3 = h2 @ W2[e].T [M,D] | scatter rows to peer combine buffers via `scatter_ptrs` |
| B11 | `api.py:2444` | `dist_grouped_gemm_dgrad_dispatch` (`:1441`) | gather grad_h3 rows from peer combine buffers (pointer table = `scatter_ptrs`) into `grad_h3_gathered` [M,D], grad_h2 = grad_h3_gathered @ W2[e] [M,F] | gather |
| B12, B14 | `api.py:2474`, `:2555` | `grouped_gemm_wgrad` (`grouped_gemm.py:640`) | W2 WGRAD [E,D,F] from grad_h3_gathered, h2; W13 WGRAD [E,2F,D] from grad_h1, x_gathered; BF16 or FP32 out, overwrite or reduce-add | none |
| B13'' | `api.py:2525` | `dist_grouped_gemm_dgrad_combine` (`:1559`) | grad_x = grad_h1 @ W13[e] [M,D] | scatter to peer dispatch buffers via `bwd_gather_ptrs` |
| B13 | `api.py:2496` | `dist_grouped_gemm_dgrad_swiglu_bwd_combine` (`:1611`) | fused SwiGLU bwd + B13'' | scatter; only used when `F % 256 == 0` |

The forward SwiGLU fusion is gated at `api.py:2053-2057` (`fused_swiglu_forward`) and the backward one at `api.py:2495` (`w2_EDF.shape[2] % native_swiglu_bwd_col_multiple() == 0`). Unfused fallbacks already exist on both (`swiglu_fwd` at `api.py:2098`, `swiglu_bwd` + `dgrad_combine` at `api.py:2513`/`:2525`), so on sm90 both gates simply evaluate false. Note `_native_swiglu_fwd_config_is_compatible` (`dist_grouped_gemm.py:144`) and `native_swiglu_bwd_col_multiple` (`native_combine_swiglu.py:46`) live in modules that import `cutlass` at load; the sm90 path must not import them to make that decision. Unused on the BF16 api path and out of scope: `dist_grouped_gemm_wgrad_dispatch`, `dist_grouped_gemm_fprop_swiglu_fwd_dispatch`, local `grouped_gemm_fprop` / `grouped_gemm_dgrad`.

The full call chains are documented in `docs/bf16_execution.md` (forward `:135-144`, backward `:299-308`) and the fused kernel design in `docs/bf16_kernel_design.md` (`:487-610`). Read both before starting.

## Hard constraints any sm90 backend must keep

1. Registered op schemas and state layouts are unchanged: `dist_moe::bf16` (`api.py:2614-2626`, meta `:2723`), `dist_moe::bf16_forward` (`:2898`, returns `(out, list[9])`, fake `:3019-3047`, runtime check of 18 saved tensors and `forward_state.numel() == 6` at `:2877-2883`), `bf16_forward_with_clip_stats` (`:3100`), `bf16_backward` (`:3358`), `bf16_backward_accumulate_` (`:3401`). `forward_state` int64[6] must not grow.
2. No host reads of device state. Every activation operand is an int64[1] device byte offset into the uint8 `activation_buffer`; row counts and `need_recompute` are device-only; recompute is device-predicated with a fixed launch sequence so CUDA graph capture and replay work. The CuTe launchers handle this by building max-capacity placeholder views and patching the base pointer on device (`dist_grouped_gemm.py:539-621`) and by an int32 `conditional_execution` flag that makes the kernel a no-op (`:924-949`). Consequence: `torch._grouped_mm` cannot be called directly on activation operands (it needs host-known tensor addresses). The sm90 kernels must take base pointer + device offset + device group sizes + device conditional flag.
3. BF16 rounding points stay where they are (h1 before SwiGLU, grad_h1 before W13 DGRAD/WGRAD). W2 WGRAD consumes h2 before grad_h1 overwrites that scratch. Peer scatter targets are fully written before the following `_ep_barrier`.
4. FakeTensor never reads contents; non-strict `make_fx` keeps the ops opaque; WGRAD destination callbacks (`dist_moe/_execution.py`) and in-place accumulation (`bf16_backward_accumulate_`) keep working.
5. Determinism: no split-K atomics. Same inputs produce bitwise-identical outputs run to run.

## Verified building blocks (installed torch 2.14.1+cu130, triton 3.8.0, nvidia-cutlass-dsl 4.8.0)

- `torch._grouped_mm` ships native sm_90a CUTLASS ptr-array grouped GEMMs for BF16 (fprop/DGRAD 2D x 3D with device `offs`; WGRAD 2D x 2D). BF16 output only (no FP32 out), no bias. Use it in tests and benchmarks as an oracle and a throughput roofline, not in the execution path (constraint 2). Its non-CUTLASS fallback calls `offs.cpu()`.
- Inductor's persistent Triton grouped-mm template `torch/_inductor/kernel/templates/triton_mm_grouped.py.jinja` (enabled for capability >= 9.0, TMA via `tl.make_tensor_descriptor`, all four layout combinations) is not usable standalone (Jinja macros) but is the reference to port into a standalone `@triton.jit` kernel.
- Triton 3.8 lowers `tl.make_tensor_descriptor` to TMA on sm90 and runs Hopper warp specialization; device-side descriptors need `triton.set_allocator` (`triton/runtime/_allocation.py:29`), whose interaction with CUDA graph capture must be verified before relying on it.
- dist_moe has no Triton GEMMs today (no `tl.dot` anywhere). Existing Triton patterns to reuse: device-offset operand loads in `dist_moe/kernels/triton/swiglu.py` (`activation_buffer_ptr` + `xy_offset_ptr`), device row count `num_recv_tokens_ptr`, `conditional_execution_ptr`; reading peer memory from int64 addresses in `_triton_dist_gather_metadata` (`dist_moe/kernels/triton/dist_dispatch_routing_kernel.py:299`); `.sys` acquire/release helpers in `dist_moe/kernels/triton/comm_utils.py`.
- CUTLASS DSL 4.8.0 has sm90 warpgroup MMA (`cute/nvgpu/warpgroup/mma.py`, requires `sm_90a`), TMA G2S, `utils/hopper_helpers.py`, `pipeline/sm90.py`; no Hopper GEMM example. TMA gather4 requires sm100, so Hopper gathers use `cp.async`.
- Upstream QuACK (`github.com/Dao-AILab/quack`, Apache-2.0, requires `nvidia-cutlass-dsl>=4.7`) has `quack/gemm_sm90.py` (`GemmSm90`: persistent warp-specialized WGMMA, cooperative/pingpong, varlen_m/varlen_k via `cu_seqlens`, `gather_A` via `A_idx` with cp.async producer warps), `varlen_utils.py`, `tile_scheduler.py` (`VarlenMTileScheduler`), `epilogue/` (`swiglu_mod`, `dswiglu_mod`, `dgated_moe_mod`). No scatter epilogue. The vendored `dist_moe/kernels/_quack` subset contains no GEMM code.

## Step 0: environment on the H200 cluster (before Step 1)

1. `git fetch upstream`; if `main` differs from `upstream/main`, ask the user which base to use.
2. Create a venv per `README.md` with `uv venv --python 3.12` and `uv pip install --torch-backend=cu130 -e . pytest ruff` (this exact command worked on the Blackwell cluster). Pins: `torch>=2.14`, `triton>=3.8,<3.10`, `nvidia-cutlass-dsl[cu13]`, `cuda-python>=13.1`. Risk: cu130 wheels need a driver that supports CUDA 13 (`nvidia-smi` driver >= 580). If the H200 driver is older, stop and ask the user.
3. Confirm `python -c "import dist_moe"` works on an H200 and record which existing tests are skipped (all kernel tests are gated on Blackwell, see Step 1.6).
4. Record GPU model, SM count (H200: 132), driver, and NVSwitch/NVLS availability (whether `initialize_multimem_barrier_workspace` in `dist_moe/_buffers.py:312` selects multimem or falls back).

## Step 1 (`feat/sm90-bf16-triton`, detailed): unfused Triton sm90 BF16 backend

Outcome: BF16 training and inference run on H200 through the unchanged public API; each Blackwell CuTe launch in the table above is replaced, on sm90 only, by standalone Triton kernels: a peer gather or scatter copy plus a grouped GEMM.

### 1.1 Backend selection
- Add a private backend resolver (for example `dist_moe/_bf16_backend.py`, `resolve_bf16_backend(device) -> Literal["cute_sm100", "triton_sm90"]`) chosen at context creation from `torch.cuda.get_device_capability(device)`: major >= 10 selects CuTe, major == 9 selects Triton, anything else raises a clear `ValueError` at `create_context`. Store the result on the context.
- A private environment override (for example `DIST_MOE_BF16_BACKEND=triton`) forces the Triton backend on Blackwell. This makes the Triton path testable on Blackwell CI and gives a CuTe-vs-Triton parity oracle. Keep it private and documented in `tests/AGENTS.md` only.
- Mirror the existing backend switch precedent at `api.py:4006-4021` (`config.block_scaled` routes to `_blockscaled._run_blockscaled`). Keep the six-call orchestration in `_Bf16Autograd.forward` / `_backward_impl` and swap only the launchers: move the lazy CuTe imports at `api.py:1937-1947` and `:2284-2303` behind the backend so the sm90 path never imports `cutlass`.
- On sm90, `fused_swiglu_forward` and the `F % 256` backward gate evaluate false without importing the CuTe modules.
- `Bf16GroupedGemmPreset` (`api.py:106-115`, validated `:484-489`) names sm100 configs: reject a non-`None` preset on the Triton backend with a clear error. `_bf16_compute_config` (`api.py:840-872`) and `kernels/config.py` presets are not used on sm90; the Triton kernels get their own small config table (1.3).
- Fused post-expert RMSNorm: keep it if `supports_fused_post_expert_rmsnorm` (`dist_moe/_postprocess.py:207`) and its sm90 branches already work; otherwise route to the unfused callback path and record it as a follow-up.

### 1.2 Peer gather and scatter kernels (new, Triton)
New module, for example `dist_moe/kernels/triton/peer_rows.py`:
- `gather_rows(ptrs_M, num_rows_1, dst=(activation_buffer, dst_offset_1), row_bytes, conditional_execution)`: for `m < num_rows`, copy one row from address `ptrs_M[m]` into `dst + m * row_bytes`; zero-fill when the pointer is 0. Peer loads use `cache_modifier=".cg"` (the CuTe gather uses `ld.cg`, `dist_grouped_gemm.py:196-208`; that comment says the choice assumes Blackwell NVLink L2 invalidation semantics, so re-check the ordering argument for H100 NVLink and document it).
- `scatter_rows(src=(activation_buffer, src_offset_1), ptrs_M, num_rows_1, row_bytes, conditional_execution)`: for `m < num_rows`, store row m to `ptrs_M[m]`, skipping pointer 0. The following `_ep_barrier` (kernel boundary plus `.sys` release/acquire) publishes the writes.
- Persistent grid over `max_num_recv` rows (host-known capacity), 16-byte vector accesses, early exit when `conditional_execution == 0`, device-offset addressing exactly like `swiglu.py`.
- Uses: F6/B4 gather `fwd_gather_ptrs` into `x_gathered`; B11 gather via `scatter_ptrs` into `grad_h3_gathered`; F8/B6 scatter h3_pre via `scatter_ptrs`; B13'' scatter grad_x_gathered via `bwd_gather_ptrs`.

### 1.3 Grouped GEMM kernel (new, Triton)
New module, for example `dist_moe/kernels/triton/grouped_gemm.py`, one persistent kernel family ported from Inductor's `triton_mm_grouped.py.jinja`:
- Operands: activations as `(activation_buffer, offset_1)` pairs or plain tensors (B11 `dy` is a real tensor, the local combine view); weights as plain tensors `[E, N, K]`; group sizes from `num_tokens_per_local_experts` (prefix sums computed on device inside the kernel or by a tiny Triton prefix kernel into a context-owned scratch tensor); total rows bounded by host-known capacity for grid sizing; `conditional_execution` no-op flag.
- Modes: FPROP `Y[M,N] = X[M,K] @ W[e,N,K]^T` (F6, F8 and recompute B4, B6); DGRAD `DX[M,K] = DY[M,N] @ W[e,N,K]` (B11, B13''); WGRAD `DW[e,N,K] = DY[M_e,N]^T @ X[M_e,K]` per expert, FP32 accumulate, BF16 or FP32 output, overwrite or reduce-add into the destination (to match `grouped_gemm_wgrad`, `grouped_gemm.py:640`, including WGRAD destination callbacks and `bf16_backward_accumulate_`). Experts with zero rows must still write zeros in overwrite mode.
- Start with pointer-based `tl.load` tiles plus Hopper software pipelining (graph-safe, simplest). Add TMA descriptors (`tl.make_tensor_descriptor`) only after verifying `triton.set_allocator` is CUDA-graph-capture safe, and only if measurements justify it.
- No split-K; deterministic. FP32 accumulation, BF16 store.
- Config: a small table keyed by (mode, N, K) with block sizes, warps, stages; tune on H200 with the profiling skill.

### 1.4 Memory planner additions
The unfused combine needs a local pre-combine buffer that does not exist today:
- Forward `h3_pre` [R, D] (R = max receive rows): in recompute and scratch-only modes it can alias `x_gathered` (same size, dead after FC13); in no-recompute mode it needs `R * D * 2` more scratch.
- Backward `grad_x_gathered` [R, D] for B13'': it cannot alias `grad_h3_gathered` (inside grad_h1's region when `D <= F`); it can alias `grad_h2` (dead after SwiGLU backward) when `D <= F`, otherwise it needs extra scratch. Backward recompute also needs an `h3_pre` slot (the grad_h2 region is free during the recompute steps).
- Both are transient within one forward or one backward, so they do not need to enter `forward_state` (constraint 1). Add a backend-gated term to the host formulas (`dist_moe/_activation_buffer.py`: `fwd_act_mem_size` `:1306`, `bwd_act_mem_size` `:1353`, `saved_act_mem_size` `:1406`, scratch peak `:1688`) and compute their device offsets from offsets the planner already emits (for example the scratch top), either in the existing planner kernels (`_triton_get_forward_plan` `_activation_buffer_planner_kernel.py:1069`, `_triton_get_backward_plan` `:1279`; read the liveness comments at `:1361-1385`) or in a small sm90-only kernel. Host sizing and device offsets must agree; extend the planner tests accordingly. Read `docs/memory_planner.md` first.
- Step 2 removes the need for these buffers on the fused path; keep them for the unfused fallback.

### 1.5 Inference
`dist_moe::bf16` (inference and no-grad) runs the same forward launchers (`api.py:1981-1999` publish path, `use_low_latency=True` routing); make sure the sm90 path covers it, including CUDA graph capture.

### 1.6 Tests
- Gating: today `DistMoeKernelTest` (`tests/test_dist_moe.py:1510-1513`) and `DistMoeTwoRankTest` (`:4514-4520`) skip unless `_is_blackwell()` (`:76-82`), with the `gb10x` marker (`pyproject.toml:57`). Add a capability helper for "BF16 backend available" (major 9 or >= 10) and a pytest marker for sm90, and move BF16-only tests to that gating per test (block-scaled tests stay Blackwell-only). Keep `tests/test_package_boundary.py` passing.
- BF16 tests to enable on H200 (names at `caf454c`): single-rank forward/backward vs `_reference_moe` (`tests/test_dist_moe.py:178`), bounded token shapes, two saved layers, clamped SwiGLU, custom-op / `make_fx` / CUDA graph replay BF16 tests, and the two-rank BF16 tests (`test_forward_and_backward_exercise_peer_paths`, bounded token shape, fully padded rank, clip statistics, flat weights, BF16 accumulates parameter grad, post-expert RMSNorm BF16 cases).
- New tests: peer gather/scatter kernels (zero-fill on pointer 0, row bound, conditional no-op, two-rank peer access); grouped GEMM vs `torch._grouped_mm` and vs FP32 PyTorch per mode, including empty experts, non-multiple-of-block rows, FP32 WGRAD and reduce-add; planner sizing for the new buffers; CUDA graph capture and replay of the full sm90 training step.
- On Blackwell, with the private override, run the same BF16 suites through the Triton backend and compare against the CuTe backend within the existing tolerances.
- Update `tests/AGENTS.md` (supported hardware lanes, the override).

### 1.7 Docs
`README.md` (platform table at `:4`, `:75`: add Hopper BF16), `docs/bf16_execution.md` (backend section: which launches change on sm90), `docs/bf16_kernel_design.md` or a new `docs/bf16_sm90_kernel_design.md` for the Triton kernels, `docs/memory_planner.md` for the new buffers.

### 1.8 Validation and evidence (H200)
In `tests/AGENTS.md` order: CPU config/planner tests, one-GPU non-VMM, real two-rank, then ruff format/lint, package and docs checks. Then performance with the `profiling` skill (`~/.claude/skills/profiling`, run its `scripts/bench_ops.py` on a cases file in `~/tmp/profiling/<investigation>/inputs/`): per-launch grouped GEMM throughput vs `torch._grouped_mm` (CUTLASS sm90 roofline) at real MoE shapes (for example DeepSeek V4 Flash routed experts: D=4096, F=2048, 256 experts, top-6; and a smaller config), gather/scatter bandwidth vs NVLink peak, and end-to-end forward+backward step time with a per-kernel breakdown (`bench_ops.py --trace` then `top_kernels.py`). Record results; they decide Step 2 priorities and Step 3 go/no-go.

## Step 2 (`feat/sm90-bf16-triton-fused`, moderate detail): fused Triton kernels

Each fusion behind its own private flag, validated against the Step 1 unfused kernels (identical arithmetic except where rows come from and go to, so outputs should match bitwise; investigate any difference), measured with the same benchmarks:
1. Gather-A: FPROP and DGRAD dispatch load A rows through the pointer table (`.cg` loads, zero-fill on 0) instead of reading `x_gathered`, and side-store the gathered tile to `x_gathered` / `grad_h3_gathered` because WGRAD and saved activations still need it (the CuTe kernel does the same, `docs/bf16_kernel_design.md:69-82`). Gathered A cannot use TMA.
2. Scatter-D: FPROP combine and DGRAD combine epilogues store each output row to its pointer-table address, removing `h3_pre` and `grad_x_gathered` from the fused path (planner keeps them only for the unfused fallback).
3. SwiGLU forward epilogue on W13 FPROP (emit h1 and h2, matching `fused_swiglu_forward` semantics and BF16 rounding points) and SwiGLU backward in the W13 DGRAD prologue (still storing grad_h1 for WGRAD), mirroring `dgrad_swiglu_bwd_combine`.
Open questions to settle with Step 1 data: whether gather-A without TMA loses more than it saves at large D; tile shapes that keep the scatter epilogue store-efficient.

## Step 3 (`feat/sm90-bf16-cute`, high level): CuTe DSL sm90 kernels

Port the fused BF16 dist grouped GEMMs to CuTe DSL on sm_90a, seeded from QuACK `GemmSm90` (vendor under `dist_moe/kernels/_quack/` with `NOTICE` and `LICENSES` updates, like the existing vendored subset): WGMMA mainloop with producer/consumer warpgroups, `VarlenMTileScheduler` for per-expert rows, `gather_A` cp.async producer warps for dispatch, the SwiGLU/dSwiGLU epilogues, plus a new scatter epilogue for combine. Integrate dist_moe's device-offset base-pointer patching (`dist_grouped_gemm.py:539-621`), `conditional_execution`, per-tile gather counters, and determinism contract, reusing `kernels/config.py`-style presets for sm90.
Go/no-go: proceed only if Step 2 measurements show a material gap to the CUTLASS roofline at production shapes (threshold to agree with the user from Step 1/2 data). Open questions: cooperative vs pingpong per mode; cluster size; register budget for the fused SwiGLU epilogue; whether to keep Triton as the fallback backend permanently. Fill these in, in this `PLAN.md`, as Steps 1 and 2 produce data.

## Landing

- Branches via `git tree`, stacked: `feat/sm90-bf16-triton` (parent `main`; holds this `PLAN.md` commit, then Step 1), `feat/sm90-bf16-triton-fused` (parent `feat/sm90-bf16-triton`, Step 2), `feat/sm90-bf16-cute` (parent `feat/sm90-bf16-triton-fused`, Step 3). Create each child only when its step starts: `git tree branch <path> <name> --parent <parent>`, with `<path>` = `/home/garrett/github/garrett361/dist_moe-<branch-without-slashes>` (for example `/home/garrett/github/garrett361/dist_moe-featsm90-bf16-triton-fused`). Propagate changes down the stack with the `git-tree-propagate` skill.
- Conventional commits with scope (for example `feat(kernels): add Triton sm90 grouped GEMM`), ending with the attribution line the session provides.
- Drop the `PLAN.md` commit before upstream PRs. Push only with explicit user approval each time.

## Risks

- Driver/toolkit on the H200 cluster may not support the cu130 stack (Step 0.2).
- Triton device-side TMA descriptors and CUDA graph capture (verify before use).
- Peer-load memory ordering on H100 NVLink differs from the Blackwell assumption noted at `dist_grouped_gemm.py:205-207`.
- Planner changes must keep host sizing and device offsets consistent; a mismatch traps inside routing.
- Triton GEMM throughput may trail CUTLASS sm90 substantially; that is what Step 3 exists to close, and Step 1/2 measurements decide whether it is worth it.
