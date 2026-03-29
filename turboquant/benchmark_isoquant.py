"""
IsoQuant vs RotorQuant: Head-to-head benchmark on the validated pipeline.

Tests both methods on:
  1. Reconstruction MSE (synthetic + real activations)
  2. Speed (quantize + dequantize latency)
  3. Compression ratio
  4. Parameter count

This uses the SAME Lloyd-Max codebooks and same evaluation as
benchmark_google_parity.py, just swapping the rotation primitive.
"""

import torch
import time
import math
import sys

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from turboquant.rotorquant import RotorQuantMSE, RotorQuantProd
from turboquant.isoquant import IsoQuantMSE, IsoQuantProd


def benchmark_mse(d, bits, n_vectors=8192, device='cuda'):
    """Compare reconstruction MSE across all methods."""
    torch.manual_seed(42)
    # Synthetic normalized vectors (same as IsoQuant paper)
    x = torch.randn(n_vectors, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    results = {}

    # RotorQuant (Cl(3,0) 3D blocks)
    rq = RotorQuantMSE(d, bits, seed=42, device=device)
    x_hat_rq, _ = rq(x)
    mse_rq = (x - x_hat_rq).pow(2).mean().item()
    results['RotorQuant'] = mse_rq

    # IsoQuant-Full (quaternion 4D blocks, double-sided)
    iq_full = IsoQuantMSE(d, bits, seed=42, mode='full', device=device)
    x_hat_full, _ = iq_full(x)
    mse_full = (x - x_hat_full).pow(2).mean().item()
    results['IsoQuant-Full'] = mse_full

    # IsoQuant-Fast (quaternion 4D blocks, single-sided)
    iq_fast = IsoQuantMSE(d, bits, seed=42, mode='fast', device=device)
    x_hat_fast, _ = iq_fast(x)
    mse_fast = (x - x_hat_fast).pow(2).mean().item()
    results['IsoQuant-Fast'] = mse_fast

    return results


def benchmark_speed(d, bits, n_vectors=8192, warmup=50, iters=200, device='cuda'):
    """Compare quantize+dequantize latency."""
    torch.manual_seed(42)
    x = torch.randn(n_vectors, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    methods = {
        'RotorQuant': RotorQuantMSE(d, bits, seed=42, device=device),
        'IsoQuant-Full': IsoQuantMSE(d, bits, seed=42, mode='full', device=device),
        'IsoQuant-Fast': IsoQuantMSE(d, bits, seed=42, mode='fast', device=device),
    }

    results = {}
    for name, model in methods.items():
        # Warmup
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        # Timed
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e6  # microseconds
        results[name] = dt

    return results


def benchmark_params(d, bits):
    """Compare parameter counts."""
    rq = RotorQuantMSE(d, bits, seed=42, device='cpu')
    iq_full = IsoQuantMSE(d, bits, seed=42, mode='full', device='cpu')
    iq_fast = IsoQuantMSE(d, bits, seed=42, mode='fast', device='cpu')

    return {
        'RotorQuant': {
            'n_groups': rq.n_groups,
            'block_size': 3,
            'mv_components': 8,
            'rotation_params': rq.rotors.numel(),
            'quantized_per_group': 3,  # only vector grade stored
        },
        'IsoQuant-Full': {
            'n_groups': iq_full.n_groups,
            'block_size': 4,
            'components': 4,
            'rotation_params': iq_full.q_L.numel() + iq_full.q_R.numel(),
            'quantized_per_group': 4,
        },
        'IsoQuant-Fast': {
            'n_groups': iq_fast.n_groups,
            'block_size': 4,
            'components': 4,
            'rotation_params': iq_fast.q_L.numel(),
            'quantized_per_group': 4,
        },
    }


def benchmark_inner_product(d, bits, n_vectors=4096, device='cuda'):
    """Compare inner product estimation accuracy (Stage 1 + Stage 2)."""
    torch.manual_seed(42)
    x = torch.randn(n_vectors, d, device=device)
    y = torch.randn(n_vectors, d, device=device)

    # True inner products
    true_ip = (x * y).sum(dim=-1)

    methods = {
        'RotorQuant': RotorQuantProd(d, bits, seed=42, device=device),
        'IsoQuant-Full': IsoQuantProd(d, bits, mode='full', seed=42, device=device),
        'IsoQuant-Fast': IsoQuantProd(d, bits, mode='fast', seed=42, device=device),
    }

    results = {}
    for name, model in methods.items():
        compressed = model.quantize(x)
        est_ip = model.inner_product(y, compressed)
        ip_mse = (true_ip - est_ip).pow(2).mean().item()
        ip_bias = (true_ip - est_ip).mean().item()
        results[name] = {'ip_mse': ip_mse, 'ip_bias': ip_bias}

    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    dims = [64, 128, 256]
    bits_list = [2, 3, 4]

    # ── MSE Benchmark ──
    print("=" * 70)
    print("  RECONSTRUCTION MSE (lower is better)")
    print("=" * 70)
    print(f"{'d':>5} {'bits':>4} | {'RotorQuant':>12} {'IsoFull':>12} {'IsoFast':>12} | {'Full vs RQ':>10} {'Fast vs RQ':>10}")
    print("-" * 70)
    for d in dims:
        for bits in bits_list:
            mses = benchmark_mse(d, bits, device=device)
            rq = mses['RotorQuant']
            full = mses['IsoQuant-Full']
            fast = mses['IsoQuant-Fast']
            # Ratio: <1 means IsoQuant is better
            r_full = full / rq if rq > 0 else float('inf')
            r_fast = fast / rq if rq > 0 else float('inf')
            print(f"{d:>5} {bits:>4} | {rq:>12.6f} {full:>12.6f} {fast:>12.6f} | {r_full:>9.3f}x {r_fast:>9.3f}x")

    # ── Speed Benchmark ──
    if device == 'cuda':
        print()
        print("=" * 70)
        print("  LATENCY (microseconds, lower is better)")
        print("=" * 70)
        print(f"{'d':>5} {'bits':>4} | {'RotorQuant':>12} {'IsoFull':>12} {'IsoFast':>12} | {'Full speedup':>12} {'Fast speedup':>12}")
        print("-" * 70)
        for d in dims:
            for bits in bits_list:
                speeds = benchmark_speed(d, bits, device=device)
                rq = speeds['RotorQuant']
                full = speeds['IsoQuant-Full']
                fast = speeds['IsoQuant-Fast']
                print(f"{d:>5} {bits:>4} | {rq:>10.0f}µs {full:>10.0f}µs {fast:>10.0f}µs | {rq/full:>11.2f}x {rq/fast:>11.2f}x")

    # ── Parameter Count ──
    print()
    print("=" * 70)
    print("  PARAMETER COUNT (d=128, bits=3)")
    print("=" * 70)
    params = benchmark_params(128, 3)
    for name, info in params.items():
        print(f"  {name}:")
        for k, v in info.items():
            print(f"    {k}: {v}")
        print()

    # ── Inner Product Accuracy ──
    print("=" * 70)
    print("  INNER PRODUCT ESTIMATION (Stage 1 + Stage 2)")
    print("=" * 70)
    print(f"{'d':>5} {'bits':>4} | {'RQ ip_mse':>12} {'Full ip_mse':>12} {'Fast ip_mse':>12} | {'RQ bias':>10} {'Full bias':>10} {'Fast bias':>10}")
    print("-" * 70)
    for d in dims:
        for bits in bits_list:
            ips = benchmark_inner_product(d, bits, device=device)
            rq = ips['RotorQuant']
            full = ips['IsoQuant-Full']
            fast = ips['IsoQuant-Fast']
            print(f"{d:>5} {bits:>4} | {rq['ip_mse']:>12.6f} {full['ip_mse']:>12.6f} {fast['ip_mse']:>12.6f} | {rq['ip_bias']:>+10.6f} {full['ip_bias']:>+10.6f} {fast['ip_bias']:>+10.6f}")

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("  If IsoQuant MSE ≈ RotorQuant MSE and speed is faster,")
    print("  it's a validated upgrade for the rotation primitive.")
    print("  The rest of the pipeline (Lloyd-Max, QJL, KV cache) stays the same.")
