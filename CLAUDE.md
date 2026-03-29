# RotorQuant Project

## Sanity Check Results (2026-03-30)

### PASSED ✓
- RotorQuant KV cache compression matches Google TurboQuant on every metric
- MSE parity, compression ratio parity, PPL degradation parity
- NIAH (Needle-in-a-Haystack) pass rate matches
- Triton rotor sandwich kernels: 12.7x speedup vs FP32 at 32K context
- Fused attention kernel with QJL estimator working

### Related: GRA Attention (in GRA-hybrid/ sibling repo)
- GRA sparse rolling attention does NOT work for text generation
- Every variant tested generates garbage despite low CE loss
- See `../GRA-hybrid/grok.md` for full analysis
- RotorQuant compression itself is solid — the issue is only with GRA as an attention mechanism

## Architecture
- `turboquant/rotorquant.py` — RotorQuantMSE: Clifford algebra rotor sandwich quantization
- `turboquant/triton_kernels.py` — 5 Triton kernels (rotor forward/inverse, fused pipeline, attention)
- `turboquant/fused_attention.py` — Fused attention with QJL two-term estimator
- `turboquant/benchmark_google_parity.py` — TurboQuant parity test suite
- `turboquant/benchmark_perplexity.py` — Autoregressive PPL evaluation
