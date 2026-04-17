"""
benchmark_m4.py
---------------
Measures single-image inference latency on Apple M4 (CPU and MPS).
Results are saved to evaluation/m4_benchmark.json and printed to stdout.

This script is intentionally separate from compare_performance.py because:
  - compare_performance.py compares ZCU104 ARM (estimated) vs FPGA (projected)
  - This script measures the MacBook M4 as a *development reference* baseline

Usage:
    python scripts/benchmark_m4.py \\
        --checkpoint /path/to/best_model.pth \\
        [--warmup 50] [--runs 500] [--input-size 96]

Output (evaluation/m4_benchmark.json):
    {
      "cpu": {"mean_ms": ..., "std_ms": ..., "fps": ..., "p95_ms": ...},
      "mps": {"mean_ms": ..., "std_ms": ..., "fps": ..., "p95_ms": ...}
    }
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.model import create_model


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_device(model: torch.nn.Module, device: torch.device,
                     input_size: int, warmup: int, runs: int) -> dict:
    """
    Run `warmup` forward passes (discarded), then time `runs` forward passes.
    Returns dict with mean_ms, std_ms, fps, p95_ms.
    """
    model = model.to(device)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    # Warmup
    print(f"    warming up ({warmup} passes) ...", end=' ', flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            # Flush MPS command buffer so warmup is truly executed
            if device.type == 'mps':
                torch.mps.synchronize()
    print("done")

    # Timed runs — one at a time, synchronise each pass
    latencies_ms = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'mps':
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'mps':
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    std_ms  = float(np.std(arr))
    p95_ms  = float(np.percentile(arr, 95))
    fps     = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        'mean_ms': round(mean_ms, 3),
        'std_ms':  round(std_ms,  3),
        'fps':     round(fps,     1),
        'p95_ms':  round(p95_ms,  3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, input_size: int) -> tuple[torch.nn.Module, dict]:
    """Load checkpoint and return (model, metadata)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Pull training args stored in the checkpoint
    stored_args = ckpt.get('args', {})
    num_classes  = len(ckpt.get('class_to_idx', {'defect': 0, 'normal': 1}))
    dropout      = stored_args.get('dropout', 0.3)

    model = create_model(
        model_type='classifier',
        num_classes=num_classes,
        input_size=input_size,
        dropout=dropout,
        freeze_backbone=False,   # All weights loaded; freeze state doesn't matter for inference
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    meta = {
        'best_acc':     ckpt.get('best_acc', 'unknown'),
        'epoch':        ckpt.get('epoch', 'unknown'),
        'class_to_idx': ckpt.get('class_to_idx', {}),
    }
    return model, meta


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='M4 inference latency benchmark')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pth')
    parser.add_argument('--input-size', type=int, default=96,
                        help='Input image size (default: 96)')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warmup passes before timing (default: 50)')
    parser.add_argument('--runs', type=int, default=500,
                        help='Timed passes per device (default: 500)')
    parser.add_argument('--output-dir', default='evaluation',
                        help='Directory for benchmark JSON (default: evaluation)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"❌  Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"📦  Loading checkpoint: {args.checkpoint}")
    model, meta = load_model(args.checkpoint, args.input_size)
    print(f"    best_acc = {meta['best_acc']}%  |  epoch {meta['epoch']}")
    print(f"    classes  = {meta['class_to_idx']}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    params   = {total_params/1e6:.2f}M")
    print()

    # ── Determine available devices ────────────────────────────────────────────
    devices_to_test = [torch.device('cpu')]
    if torch.backends.mps.is_available():
        devices_to_test.append(torch.device('mps'))
        print("✅  MPS (Apple GPU) available — will benchmark both CPU and MPS")
    else:
        print("ℹ️   MPS not available — benchmarking CPU only")
    print()

    # ── Run benchmarks ──────────────────────────────────────────────────────────
    results = {}
    for device in devices_to_test:
        label = device.type
        print(f"⏱️   Benchmarking [{label.upper()}]  "
              f"({args.warmup} warmup + {args.runs} timed, "
              f"input {args.input_size}×{args.input_size})")
        stats = benchmark_device(
            model, device, args.input_size, args.warmup, args.runs
        )
        results[label] = stats
        print(f"    mean latency : {stats['mean_ms']:.3f} ms  "
              f"(±{stats['std_ms']:.3f} ms)")
        print(f"    P95 latency  : {stats['p95_ms']:.3f} ms")
        print(f"    throughput   : {stats['fps']:.1f} FPS")
        print()

    # ── Save JSON ───────────────────────────────────────────────────────────────
    output = {
        'model': {
            'checkpoint': args.checkpoint,
            'best_acc_pct': meta['best_acc'],
            'epoch': meta['epoch'],
            'params_M': round(total_params / 1e6, 3),
            'input_size': args.input_size,
        },
        'benchmark_config': {
            'warmup_runs': args.warmup,
            'timed_runs':  args.runs,
            'batch_size':  1,
        },
        **results,
    }

    out_path = os.path.join(args.output_dir, 'm4_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"💾  Results saved to {out_path}")
    print()

    # ── Paper-ready summary ─────────────────────────────────────────────────────
    print("=" * 55)
    print("  PAPER REFERENCE NUMBERS  (M4 MacBook, dev machine)")
    print("=" * 55)
    for label, stats in results.items():
        print(f"  {label.upper():<6}  {stats['mean_ms']:7.2f} ms   "
              f"{stats['fps']:7.1f} FPS   P95={stats['p95_ms']:.2f} ms")
    print()
    print("  Note: These are M4 development-machine reference values.")
    print("  The paper's CPU column = ZCU104 ARM Cortex-A53 (estimated).")
    print("  Add M4 numbers as a footnote for transparency.")


if __name__ == '__main__':
    main()
