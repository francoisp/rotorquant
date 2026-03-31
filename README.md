# RotorQuant: KV Cache Compression for LLMs

Drop-in KV cache quantization using rotation-based decorrelation. **4-5x compression**, **+19% PPL** at 4-bit, **367 tok/s** decode on RTX 5090 via [llama.cpp](https://github.com/johndpope/llama-cpp-turboquant/tree/feature/planarquant-kv-cache). Works on CUDA and Apple Silicon.

## Results

### Decode Speed (llama.cpp, Qwen2.5-3B Q4_K_M)

| Hardware | Cache K | Decode tok/s | Prefill tok/s | PPL |
|----------|---------|-------------|---------------|-----|
| **RTX 5090** | planar3 | **367** | **23,600** | 9.98 |
| **RTX 5090** | iso3 | **367** | **23,600** | 9.98 |
| RTX 5090 | FP16 | 356 | 20,800 | 10.03 |
| M4 Mac Mini | planar3 | 48.3 | 554 | 9.98 |
| M4 Mac Mini | FP16 | 47.4 | 518 | 9.98 |

**Llama 3.1 8B (RTX 5090):**

| Cache K | Decode tok/s | Prefill 2K | PPL |
|---------|-------------|------------|-----|
| planar3 | **239** | **13,030** | **8.44** |
| iso3 | **239** | **13,050** | **8.44** |
| FP16 | 229 | 9,360 | 8.44 |

With deferred quantization, planar3/iso3 are **3% faster than FP16** (K-cache stays F16 during prefill, no dequant overhead in flash attention) and **PPL matches FP16 exactly**.

Reproduce:
```bash
# CUDA
cd /tmp && git clone https://github.com/johndpope/llama-cpp-turboquant.git
cd llama-cpp-turboquant && git checkout feature/planarquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/llama-bench -m model.gguf -ngl 99 -ctk planar3 -ctv f16 -p 512 -n 128

# Metal (Apple Silicon)
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/llama-bench -m model.gguf -ngl 99 -ctk planar3 -ctv f16 -p 512 -n 64
```

### Perplexity (Qwen2.5-3B, wikitext-2, post-prefill, CUDA/Triton)

| Method | 3-bit PPL | 4-bit PPL | vs FP16 (7.59) |
|--------|-----------|-----------|----------------|
| **IsoQuant** | 12.35 | **9.03** | **+19%** |
| **PlanarQuant** | **10.12** | 9.56 | **+33% / +26%** |
| RotorQuant | 12.22 | 10.03 | +61% / +32% |

```bash
python -m turboquant.benchmark_google_parity --model Qwen/Qwen2.5-3B-Instruct --bits 3 4
```

### VRAM Savings (3-bit, 4.9x compression)

| Context | FP16 KV | Compressed | Saved |
|---------|---------|------------|-------|
| 8K | 288 MB | 59 MB | **230 MB** |
| 32K | 1,152 MB | 234 MB | **918 MB** |
| 65K | 2,304 MB | 469 MB | **1.8 GB** |

Needle-in-Haystack passes at 8K, 32K, and 65K context.

## Quick Start

### llama.cpp (recommended — fastest)

```bash
git clone https://github.com/johndpope/llama-cpp-turboquant.git
cd llama-cpp-turboquant && git checkout feature/planarquant-kv-cache

# CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Metal (Apple Silicon)
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Run
./build/bin/llama-server -m model.gguf --jinja -ngl 99 \
    --cache-type-k planar3 --cache-type-v f16 --host 0.0.0.0 --port 8080

# Perplexity
pip install datasets
python3 -c "from datasets import load_dataset; open('/tmp/wiki.txt','w').write('\n'.join(load_dataset('wikitext','wikitext-2-raw-v1',split='test')['text']))"
./build/bin/llama-perplexity -m model.gguf -f /tmp/wiki.txt -ngl 99 -c 512 --chunks 20 \
    --cache-type-k planar3 --cache-type-v f16
```

Cache types: `planar3`, `iso3`, `planar4`, `iso4` (ours) + `turbo3`, `turbo4` (TheTom's WHT)

### Python/Triton (research)

```bash
pip install -e . && pip install triton
```

```python
from turboquant import IsoQuantMSE, PlanarQuantMSE

# IsoQuant: best 4-bit quality (PPL 9.03)
iq = IsoQuantMSE(d=128, bits=4, mode='fast', device='cuda')
x_hat, indices = iq(x)

# PlanarQuant: best 3-bit quality (PPL 10.12)
pq = PlanarQuantMSE(d=128, bits=3, device='cuda')
x_hat, indices = pq(x)
```

## How It Works

Rotation decorrelates KV cache vectors before scalar quantization:

1. **Normalize** → store norms separately
2. **Rotate** via block transform (breaks coordinate correlations)
3. **Quantize** each coordinate to Lloyd-Max centroids
4. **Inverse rotate** to reconstruct

| | Block | FMAs (d=128) | Params | Quality |
|---|-------|-------------|--------|---------|
| TurboQuant | Dense d×d | 16,384 | 16,384 | baseline |
| **IsoQuant** | **4D quaternion** | **512** | **128** | **1.0x** |
| **PlanarQuant** | **2D Givens** | **256** | **128** | **1.0x** |

**Deferred quantization**: K-cache allocates as FP16 during prefill (zero error compounding). Decode tokens get quantized on insertion. This gives 3x better PPL than roundtrip quantization — and in llama.cpp, the F16 prefill actually makes decode **faster** than FP16 baseline.

## Benchmarks

```bash
python -m turboquant.benchmark_google_parity          # PPL (post-prefill)
python -m turboquant.benchmark_perplexity --bits 3 4   # PPL (roundtrip)
python -m turboquant.benchmark_triton                  # Triton kernel speed
python -m turboquant.poc_high_context --backend planar  # High-context generation
```

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — Google's KV cache compression
- [IsoQuant / PlanarQuant](https://github.com/ParaMind2025/isoquant) — ParaMind2025
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — llama.cpp fork with TurboQuant
- [QJL](https://arxiv.org/abs/2406.03482) — 1-bit quantized JL transform

## Citation

```bibtex
@article{pope2026rotorquant,
  title={RotorQuant: Clifford Algebra Vector Quantization for LLM KV Cache Compression},
  author={Pope, John D.},
  year={2026},
  url={https://github.com/scrya-com/rotorquant}
}
```

MIT License
