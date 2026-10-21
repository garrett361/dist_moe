# Plan: DeepSeek V4 clamped SwiGLU support in dist_moe (`swiglu_beta`)

## Context

dist_moe (`meta-pytorch/dist_moe`; this fork is `garrett361/dist_moe`, remotes `origin` = fork, `upstream` = meta-pytorch) supports two expert activations via `dist_moe.api.Config.activation`: `"swiglu"` and `"swiglu_clamped"`. The clamped form is the GPT-OSS formula:

```
g = min(gate, L);  u = clamp(up, -L, L);  out = g * sigmoid(alpha * g) * (u + 1)
```

DeepSeek V4 Flash and V4.1 Flash require a different clamped form, applied to both routed and shared experts:

```
out = silu(min(gate, L)) * clamp(up, -L, L),   L = swiglu_limit = 10.0
```

Setting `swiglu_alpha=1.0` turns `g*sigmoid(g)` into `silu(g)`, but the `+1` on the up branch is hardcoded in every forward and backward implementation (Triton and CuTe DSL, BF16 and block-scaled). So no dist_moe config reproduces DeepSeek V4. The clamp bounds themselves already match (gate clamped above only, up clamped both sides, gradient zeroed strictly outside the bound). This gap was confirmed by reading the code at `main` = `caf454c`.

vllm generalizes both models as `gate * sigmoid(alpha * gate) * (up + beta)` (`SiluAndMulWithClamp`): GPT-OSS is `beta=1`, DeepSeek V4 is `alpha=1, beta=0`. We adopt the same parametrization.

Decisions already made with the user:
- API: add a float config field `swiglu_beta` (default `1.0`, preserving current GPT-OSS behavior bit-for-bit). DeepSeek V4 callers set `activation="swiglu_clamped", swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.0`. Beta is a compile-time constant exactly like alpha and limit.
- Scope: every clamped path (BF16 Triton and CuTe fused, block-scaled staged / Mega / chunked Mega, inference interleaved epilogue, standalone quantize entry points).
- PR description (`PR.md`, written with the `/pr-description` skill) links to three golden references at their latest released versions: vllm, transformers, and DeepSeek's own `inference/model.py` on the Hugging Face hub.
- Use this exercise to improve the user's `profiling` skill (CuTe DSL and Blackwell content, tooling) with everything found that is unexpected and undocumented.

Hardware constraint: the planning cluster is H200 (sm90). dist_moe's CuTe kernels need Blackwell (sm100). The user will reclone this repo, with this plan committed on the branch, on a Blackwell cluster to run anything GPU-heavy. Phase split below marks what runs where.

## Reference semantics (verified)

DeepSeek official reference, `inference/model.py` in the HF checkpoint repos (`deepseek-ai/DeepSeek-V4-Flash-0731` `Expert.forward` ~L600-610; `deepseek-ai/DeepSeek-V4.1-Flash` `Expert` ~L830-848):

```python
gate = self.w1(x).float(); up = self.w3(x).float()
if self.swiglu_limit > 0:
    up = torch.clamp(up, min=-L, max=L); gate = torch.clamp(gate, max=L)
x = F.silu(gate) * up
```

- `swiglu_limit: 10.0` in every checkpoint config.json (V4-Flash-0731, V4-Flash-Base, V4.1-Flash under nested `text_config`). V4 and V4.1 use the identical formula. Routed and shared experts both clamp. Gate/up layout: halves, gate (w1) first, matching dist_moe's W13 `[E, 2F, K]` with gate in the first F rows.
- The official reference clamps in fp32. HF transformers clamps in the native dtype (bf16); vllm's CUDA and Triton kernels clamp in fp32. dist_moe kernels already compute the activation in fp32 registers, matching the official reference.
- vllm (upstream SHA `7d1d8660d620149bc4eadf795cbb80a1f884ddc2`, contained in tag `v0.32.0`): `vllm/model_executor/layers/activation.py` `SiluAndMulWithClamp` ~L208-263; `vllm/models/deepseek_v4/nvidia/model.py` ~L132-190 (shared expert), L705, L889 (routed); `vllm/model_executor/layers/fused_moe/activation.py` ~L166-195, L293-311, L350-356; `vllm/model_executor/layers/fused_moe/utils.py` ~L509-626 (pad-aware Triton clamp kernel). Line numbers must be re-verified at the pinned tag.
- transformers (`/home/garrett/github/garrett361/transformers`, now at upstream main `8563d78e`, no tags fetched yet): `src/transformers/models/deepseek_v4/configuration_deepseek_v4.py` L155 `swiglu_limit: float = 10.0`; `modeling_deepseek_v4.py` ~L1005-1061 (`DeepseekV4MLP` shared expert, `DeepseekV4Experts._apply_gate` routed, `gate_up.chunk(2, dim=-1)`).

## Phase 0: branch setup (planning cluster)

1. `git fetch upstream` in `/home/garrett/github/garrett361/dist_moe` (never fetched before). Compare `main` against `upstream/main` (`git log --oneline main...upstream/main`). If they differ, stop and ask the user which base to use; the PR targets `upstream/main`.
2. Create the branch with `git tree branch <path> <name> --parent <parent>` (parent `main` if it equals `upstream/main`). Proposed name `feat/swiglu-beta`, path `/home/garrett/github/garrett361/dist_moe-featswiglu-beta`.
3. Commit this file as `PLAN.md` in a first commit (`chore: add implementation plan`) so the user can push the branch to `origin` (only with explicit user approval) and reclone on Blackwell. Drop this commit before opening the upstream PR: AGENTS.md requires public-release-clean content.

## Phase 1: implementation (can be written on H200; grouped into commits)

Conventions (from `AGENTS.md`): Google-style docstrings on every non-kernel callable; inline comments only for non-obvious invariants; preserve kernel arithmetic, launch geometry, and op order; no em-dashes in files.

How beta is threaded:
- `canonical_swiglu_clamp(clamped, alpha, limit, beta) -> (bool, float, float, float)` in both `dist_moe/formats.py:112` and `dist_moe/kernels/triton_quantization/formats.py:127`, with new `SWIGLU_CLAMP_BETA_DEFAULT = 1.0`. Unclamped returns all defaults (single cache entry for plain SwiGLU). Always return `float(beta)` so Triton never promotes through i32.
- Append `beta` directly after `limit` in every `(clamped, alpha, limit)` sequence: Constexpr params, NamedTuples, kernel attributes, `CombineSwigluQuantParams`, Triton `tl.constexpr`.
- Internal `@cute.jit` / `@triton.jit` helpers whose `limit` has no default get `beta` with no default, so a missed positional caller fails at trace time instead of silently using 1.0.
- Replace `(u + 1)` with `(u + beta)` in exactly three roles per kernel: forward product, the dgate factor, and the H2 recompute. Rename `u_plus_one*` to `u_plus_beta*`. Backward math (`s = sigmoid(alpha g)`): `dz/dgate = (u+beta) * (s + alpha g s (1-s))`, zeroed where `gate > L`; `dz/dup = g s`, zeroed where `|up| > L` (independent of beta).
- Bitwise identity at beta=1: CuTe currently uses `u + Float32(_FP32_ONE)` (`_FP32_ONE: Constexpr[float] = 1.0`, `dist_moe/kernels/_quant_conversion.py:51`); a Constexpr beta of 1.0 emits the same constant and fadd. Triton currently uses literal `1.0`; a `BETA: tl.constexpr` of 1.0 lowers identically. `_build_swiglu_clamp_tag` (`dist_moe/kernels/blockscaled_quantize.py:211`) appends a beta tag only when `beta != 1.0`, so GPT-OSS kernel names are unchanged. This identity is a claim to verify in Phase 3, not assume.
- No registered op schema changes: `dist_moe::bf16`, `dist_moe::bf16_forward`, `dist_moe::block_scaled_forward` carry only `context_id` and read `context.config` inside. `dist_moe/_blockscaled_ops.py` functions are plain Python wrappers.

### Commit A: `feat(kernels): parametrize clamped SwiGLU up-projection offset`
No behavior change (all defaults 1.0). Sites (line numbers at `caf454c`; re-locate by function name):
1. Both `formats.py` files as above.
2. `dist_moe/kernels/blockscaled_quantization_common.py`: `_swiglu_clamped_fwd_f32` (~889), `_swiglu_clamped_bwd_f32` (~903-926), `_swiglu_clamped_bwd_f32x2` (~997-1037), dispatchers `_apply_block_scaled_quant_producer` (~931), `_dxy_producer` (~964), `_dxy_fwd_producer` (~981), `_f32x2` variants (~1091, ~1115).
3. `dist_moe/kernels/triton/swiglu.py`: `BETA: tl.constexpr` on `_triton_swiglu_fwd` (~64, math ~203) and `_triton_swiglu_bwd` (~513, math ~649); `beta` on host `swiglu_fwd` (~272) and `swiglu_bwd` (~671); all launches.
4. `dist_moe/kernels/triton_quantization/blockscaled_quantize.py`: producer helpers (~771, ~804, ~832, ~866), kernels (~1958, ~2182, ~2393, ~3128), host functions `quantize_swiglu_bwd_dxy_fwd_col`, `quantize_block_scaled`, `quantize_block_scaled_axes`.
5. `dist_moe/kernels/blockscaled_quantize.py` (CuTe): `_build_swiglu_clamp_tag`, `jit_cache` compile signatures and kernel `__init__`s, entry points `_quantize_nvfp4_per_token` (~5080), ~5225, ~5460.
6. `dist_moe/kernels/_swiglu_quant.py`: `_load_swiglu_fwd_nvfp4_block`, `_load_swiglu_fwd_fp8_tile_values`, `_load_swiglu_bwd_dxy_h2_as_f32x2`, `_load_swiglu_bwd_fp8_tile_values`, `_quantize_swiglu_fwd_fp8_tile`, `_quantize_swiglu_bwd_dxy_fp8_tile`.
7. `dist_moe/kernels/params.py` `CombineSwigluQuantParams` (~398-401) gains `combine_swiglu_beta`; filled in `params_from_kernel` (`dist_blockscaled_entrypoints.py` ~1240/1260); used in `dist_moe/kernels/combine_swiglu_quant.py` (~279, ~452).
8. `dist_moe/kernels/swiglu_epilogue.py`: `_load_interleaved_swiglu_values` and the `store_/quantize_interleaved_swiglu_*` and `blockscaled_epilog_postprocess_store` chain.
9. BF16 CuTe: `dist_moe/kernels/swiglu_fragment.py` (`SwigluH2Epilogue`, `h1_swiglu_epilogue`, `_swiglu_words`); `dist_moe/kernels/native_combine_swiglu.py` (`_native_swiglu_bwd_words` and callers); `dist_moe/kernels/dist_grouped_gemm_kernel.py` (`__init__`, launch ~2056); `dist_moe/kernels/dist_grouped_gemm.py` (`_get_dist_kernel` cache key, `_fused_dist_grouped_gemm_impl` compile key, `dist_grouped_gemm_fprop_dispatch`, `dist_grouped_gemm_dgrad_swiglu_bwd_combine`).
10. Block-scaled launchers: `blockscaled_grouped_gemm.py` `_set_swiglu_clamp_config` (writes `COMBINE_SWIGLU_BETA` when clamped); `dist_blockscaled_grouped_gemm_kernel.py` (`__init__`, `from_config` reading beta with default 1.0); `dist_blockscaled_grouped_gemm.py`; `mega_blockscaled_grouped_gemm.py` (incl. compile key ~1704); `chunked_mega_blockscaled_grouped_gemm.py` (`swiglu_clamp` tuple annotation/default ~680 becomes a 4-tuple).
11. Audit gate: `rg -nw 'limit|LIMIT|swiglu_limit|combine_swiglu_limit' dist_moe` and `rg -n 'canonical_swiglu_clamp|_set_swiglu_clamp_config|_build_swiglu_clamp_tag' dist_moe`: every clamp-related hit has a matching beta. `rg -n 'u_plus_one|\+ 1\.0\)|_FP32_ONE\)' dist_moe/kernels` shows no clamped-path leftovers.

### Commit B: `feat(api): add swiglu_beta for DeepSeek V4 clamped SwiGLU`
1. `dist_moe/api.py`: field `swiglu_beta: float = SWIGLU_CLAMP_BETA_DEFAULT` after `swiglu_limit` (~403); docstring formula (~371-375) becomes `... * (clamp(up, -limit, limit) + beta)` and documents beta=1 as GPT-OSS and `alpha=1, beta=0, limit=10` as DeepSeek V4; validation (~499-512) adds `math.isfinite(self.swiglu_beta)` with message "clamped SwiGLU requires finite swiglu_beta" (any finite sign allowed); thread `beta` at the forward (~2091-2103), recompute (~2379-2381), and DGRAD (~2508-2519) call sites.
2. `dist_moe/_blockscaled.py`: autograd Function forward signature (~1361) and docstring, `ctx.swiglu_beta`, all call sites (~1745, 1832, 2175, 2233, 2457, 2483), `backward` returns `(None,) * 38` instead of 37 (~2549), positional `.apply(...)` (~2851), config reads (~3439, 3797, 4561).
3. `dist_moe/_blockscaled_ops.py`: four wrappers and their docstrings.
4. Docs (fold into this commit or a separate `docs:` commit): `README.md` "Clamped SwiGLU" (~644-662, add a DeepSeek V4 config snippet), `docs/bf16_execution.md` (~246-250), `docs/bf16_kernel_design.md` (~178-180), `docs/mxfp8_execution.md` (~376-379), `docs/mxfp8_kernel_design.md` (~131), `docs/nvfp4_kernel_design.md` (~101, ~120).

Not changed: `dist_moe/_execution.py` SwiGLU clip stats (they count pre-activations only). Pre-existing plain-only paths (`dist_grouped_gemm_fprop_swiglu_fwd_dispatch`, `_blockscaled_ops.dist_blockscaled_grouped_gemm_fprop_dispatch`, `_quantization.swiglu_fwd_quantize/swiglu_bwd_quantize`) are unreachable with a clamped config; flag them in the PR, do not fix.

### Commit C: `test: cover DeepSeek V4 clamped SwiGLU`
All in `tests/test_dist_moe.py`:
- `_reference_moe` (~179-243): add `swiglu_beta: float = 1.0`, use `(up_f32 + swiglu_beta)`, and switch the gate clamp to `torch.clamp(gate_f32, max=swiglu_limit)` (matches the kernel's pass-through at `gate == L`; `torch.minimum` splits the gradient at ties). Thread beta through `_create_case` (~2175), `_assert_mxfp8_training_effects_preserve_eager_bitwise` (~3695), `_run_nvfp4_inference_cuda_graph_replay` (~4175), `_run_two_rank_numerics` (~4587).
- Make the clamp actually fire. Today `_create_case` yields pre-activations ~N(0,1), so with limit 5 the clamp essentially never triggers and the masks are untested. Add a `preactivation_scale` knob scaling w13 (about 8 for L=10, about 4 for L=5) and assert a non-trivial clamped fraction via `ExecutionOptions.swiglu_clip_stats_out_3`. Do not loosen tolerances; if new clamp-active inputs fail, investigate.
- Parameter cases `(alpha, beta, limit)` in `{(1.5, 1.0, 5.0), (1.0, 0.0, 10.0)}` as subtests of `test_bf16_clamped_swiglu_matches_reference` (~4369; plus a subtest with intermediate dim not a multiple of 256 to reach Triton `swiglu_bwd`, and one with clip stats on to reach Triton `swiglu_fwd`), `test_mxfp8_clamped_swiglu_matches_reference` (~5143), the effects-bitwise tests (~3985, ~3992), the NVFP4 graph matrix (~4248). New: a two-rank MXFP8/NVFP4 inference case with `fuse_swiglu=True` and V4 clamping (covers `swiglu_epilogue.py` and chunked Mega; no clamped interleaved-inference test exists today).
- CPU validation: extend `test_rejects_invalid_clamped_swiglu_parameters` (~765) with `swiglu_beta` in `{nan, inf, -inf}`; assert `0.0` and `-0.5` accepted, `Config().swiglu_beta == 1.0`, and `canonical_swiglu_clamp` behavior for both formats modules.
- Default-vs-explicit `swiglu_beta=1.0` produces bitwise-equal outputs and gradients.
- Any-GPU Triton test: `kernels.triton.swiglu.swiglu_fwd/bwd` with V4 params and scaled inputs against torch autograd, gated like `DistMoeBufferTest` (CUDA available), not `_is_blackwell`.

## Phase 2: validation on H200 (before recloning)

Follow `tests/AGENTS.md` validation order. Runnable here: `DistMoeConfigTest` (CPU), ruff/format/lint, `StaticPackageBoundaryTest`, docs checks, direct Triton `swiglu_fwd/bwd` numerics, direct Triton block-scaled quant SwiGLU producer (has sm90 branches), Triton TTIR/PTX identity main-vs-branch at beta=1 (separate `TRITON_CACHE_DIR` per tree), Python checks of `torch.ops.dist_moe.*` schema strings. Also try whether CuTe DSL can emit sm_100a SASS on an sm90 host (unverified); if it can, the CuTe SASS identity diff moves here. Record the answer in the profiling skill either way.

## Phase 3: validation and evidence on Blackwell (after recloning)

1. Full kernel test matrix per `tests/AGENTS.md`: one-GPU non-VMM, two-rank, graph/make_fx, effects-bitwise.
2. SASS identity main vs branch at beta=1: run seeded BF16, MXFP8 staged/Mega, NVFP4 inference (incl. `fuse_swiglu`) fwd+bwd with `CUTE_DSL_KEEP=sass` and a pre-created `CUTE_DSL_DUMP_DIR` per tree (the dump dir must exist; keeping sass disables the JIT cache). Diff per kernel (`_cute_dist_grouped_gemm_*swiglu*`, `_cute_mega_blockscaled_grouped_gemm_combine_bprop*`, chunked Mega forward, staged swiglu combine kernels, `_cute_quantize_*`). Expect zero diff.
3. Bitwise outputs main vs branch: out-of-tree script in `~/tmp/` that saves outputs and gradients per pipeline at the default config and compares with `torch.equal`. Also compare `context.memory_plan` and `torch.cuda.max_memory_allocated`.
4. Cost of beta=0 vs beta=1: SASS diff (expect only the constant changing; `u + 0.0` keeps its FADD under IEEE), then `triton.testing.do_bench_cudagraph` at DeepSeek V4 Flash routed-expert shapes (read `hidden_size`, `moe_intermediate_size` (2048), `n_routed_experts` (256), top-k from the `deepseek-ai/DeepSeek-V4-Flash-0731` config.json), 1 and 2 ranks: BF16 fwd+bwd plus isolated fprop dispatch and DGRAD combine, Triton swiglu fwd/bwd, MXFP8 staged/Mega step, NVFP4 Mega inference. Report deltas against run-to-run noise per the `profiling` skill's A/B rules. Add `const_expr(beta == 0.0)` elision only if a cost is measurable.

## Phase 4: profiling skill improvements (separate repo)

The skill lives at `/home/garrett/github/garrett361/dotfiles/agents_global/skills/profiling` (symlinked as `~/.claude/skills/profiling`); commit there, separately from dist_moe. Every CuTe DSL recipe in `tools/cute-dsl.md` was verified only on H200. During Phases 2-3, record every unexpected or undocumented finding, verified, with the hardware it was verified on. Known candidates:
- New `projects/dist_moe.md`: explicit `cute.compile` executor cache (`_compile_or_get`, `dist_moe/kernels/blockscaled_grouped_gemm.py:195-238`), kernel name prefixes (`dist_moe/kernels/_kernel_name_prefix.py`, `ncu -k regex:_cute_`), `record_function` ranges (`dist_moe_*` in `api.py`, `moe_*` in `_blockscaled.py`), the `CUTE_DSL_OPT_LEVEL` config-dict key and `--opt-level=1 --ptxas-options='--Ofast-compile=min'` options, CUDA-graph capture tests, IKET shim (`dist_moe/kernels/_iket_shim.py`), no in-repo bench scripts.
- `tools/cute-dsl.md` Blackwell section: re-verify dumps, `CUTE_DSL_LINEINFO`, cache dir on sm100; resolve the cache env var discrepancy (skill says `CUTE_DSL_CACHE_DIR`; dist_moe's `_iket_shim.py:33` says `CUTEDSL_CACHE_ENABLED=0`) against the installed DSL source; cross-arch SASS emission answer from Phase 2.
- `tools/sass.md` / `scripts/sass_inventory.py`: tcgen05 / TMEM / UMMA SASS opcode taxonomy beyond `UTCHMMA`, as observed in dist_moe kernels.
- Epilogue-fusion A/B method: SASS diff of the epilogue region, `do_bench_cudagraph` of fused kernel vs unfused baseline, IKET marks around the epilogue.
- ncu is unavailable on B300 (admin-only counters); document the fallback used in Phase 3.
- Clean stray `.ruff_cache/` and `__pycache__/` from the skill directory only if untracked and the user agrees.

## Phase 5: PR description

Use the `/pr-description` skill to write `PR.md` (untracked, not committed) in the branch worktree. Short prose: the gap, the `swiglu_beta` parametrization (matches vllm's alpha/beta), bitwise-unchanged default, evidence summary (tests, SASS identity, beta=0 cost). Static GitHub/HF permalinks to the latest released versions, with line numbers re-verified at the pinned ref:
- vllm: latest release tag `v0.32.0` (contains `7d1d866`; confirm no newer non-rc release with `git -C /home/garrett/github/garrett361/vllm-main fetch upstream --tags`). Link by commit SHA of the tag.
- transformers: run `git -C /home/garrett/github/garrett361/transformers fetch upstream --tags`, pick the newest release tag containing `src/transformers/models/deepseek_v4/`, link by that tag's commit SHA.
- DeepSeek: `huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/<revision>/inference/model.py` and the V4.1-Flash equivalent, pinned to the hub revision SHA (HF repos have no releases), plus each repo's `config.json` for `swiglu_limit: 10.0`.
Keep all text public-release clean (no private project names, internal paths, or cluster names).

## Verification summary

- H200: CPU config tests, lint/package/docs checks, Triton numerics and PTX identity pass.
- Blackwell: full test matrix passes with V4 cases where the clamp demonstrably fires; CuTe SASS and outputs identical to `main` at the default; beta=0 cost within noise.
- Audit greps from Commit A step 11 come back clean.
- Profiling skill commits in dotfiles each cite the hardware they were verified on.

## Risks

- A missed positional site silently runs with beta=1: mitigated by no-default helper params, the canonical tuple arity change, the audit greps, and V4 numerics on every path.
- Clamp-active tests may expose pre-existing boundary or tolerance issues in the GPT-OSS path; investigate, never weaken assertions.
- At beta=0 the result differs from `silu(g)*u` only in the sign of zero when `up = -0.0`; benign.
- `swiglu_beta` with `activation="swiglu"` is silently ignored, matching how alpha/limit behave today.
- Shared experts live outside dist_moe; callers apply the same clamp there.
