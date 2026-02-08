"""
Benchmark decoding speed with vs without KV cache.

Uses MultiHeadAttention from the mla package to compare autoregressive
decoding performance. Also calculates full-model KV cache memory for
DeepSeek-V2 scale.

Generates: figures/kv_cache_benchmark.png
"""

import os
import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mla import MultiHeadAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def benchmark_decoding_with_cache(model, d_model, num_tokens, num_warmup=2, num_runs=3):
    """Benchmark decoding with KV cache - only pass new token each step."""
    model.eval()
    times = []

    for run in range(num_warmup + num_runs):
        model.reset_cache()
        torch.cuda.empty_cache() if device.type == 'cuda' else None

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.perf_counter()

        with torch.no_grad():
            for t in range(num_tokens):
                new_token = torch.randn(1, 1, d_model, device=device)
                _ = model(new_token, use_cache=True)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.perf_counter() - start_time

        if run >= num_warmup:
            times.append(elapsed)

        model.reset_cache()

    return np.mean(times), np.std(times)


def benchmark_decoding_no_cache(model, d_model, num_tokens, num_warmup=2, num_runs=3):
    """Benchmark decoding without KV cache - pass full sequence each step."""
    model.eval()
    times = []

    for run in range(num_warmup + num_runs):
        torch.cuda.empty_cache() if device.type == 'cuda' else None

        sequence = torch.randn(1, 1, d_model, device=device)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.perf_counter()

        with torch.no_grad():
            for t in range(num_tokens):
                _ = model(sequence)
                sequence = torch.cat([sequence, torch.randn(1, 1, d_model, device=device)], dim=1)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.perf_counter() - start_time

        if run >= num_warmup:
            times.append(elapsed)

    return np.mean(times), np.std(times)


def measure_kv_cache_memory(d_model, n_heads, seq_len, dtype=torch.float16):
    """Calculate KV cache memory in MB."""
    d_h = d_model // n_heads
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    kv_cache_bytes = 2 * n_heads * seq_len * d_h * bytes_per_elem
    return kv_cache_bytes / (1024**2)


def kv_cache_memory_full_model(seq_len, n_layers, n_heads, d_h, dtype_bytes=2):
    """
    Calculate KV cache memory for an entire model.

    Args:
        seq_len: Sequence length (context window)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads per layer
        d_h: Dimension per head
        dtype_bytes: Bytes per element (2 for fp16/bf16)

    Returns:
        Memory in GB
    """
    bytes_per_token = 2 * n_layers * n_heads * d_h * dtype_bytes
    total_bytes = bytes_per_token * seq_len
    return total_bytes / (1024**3)


if __name__ == "__main__":
    # ============== Configuration ==============

    d_model = 768
    n_heads = 12

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    if device.type == 'cpu':
        seq_lengths = [128, 256, 512, 1024]
        print(f"CPU detected - limiting to {seq_lengths[-1]} tokens")

    print(f"\nBenchmarking with d_model={d_model}, n_heads={n_heads}")
    print(f"Sequence lengths: {seq_lengths}")
    print("=" * 60)

    # ============== Run Benchmarks ==============

    model_with_cache = MultiHeadAttention(d_model, n_heads).to(device)
    model_no_cache = MultiHeadAttention(d_model, n_heads).to(device)

    times_with_cache = []
    times_with_cache_std = []
    times_no_cache = []
    times_no_cache_std = []

    for num_tokens in seq_lengths:
        print(f"\nGenerating {num_tokens} tokens...")

        mean_time, std_time = benchmark_decoding_with_cache(model_with_cache, d_model, num_tokens)
        times_with_cache.append(mean_time)
        times_with_cache_std.append(std_time)
        print(f"  With KV cache:    {mean_time:.4f}s +/- {std_time:.4f}s")

        mean_time, std_time = benchmark_decoding_no_cache(model_no_cache, d_model, num_tokens)
        times_no_cache.append(mean_time)
        times_no_cache_std.append(std_time)
        print(f"  Without KV cache: {mean_time:.4f}s +/- {std_time:.4f}s")

        speedup = times_no_cache[-1] / times_with_cache[-1]
        print(f"  Speedup: {speedup:.1f}x")

        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # ============== Calculate Memory ==============

    memory_with_cache = [measure_kv_cache_memory(d_model, n_heads, seq_len) for seq_len in seq_lengths]

    # ============== Plotting ==============

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Time comparison
    ax1 = axes[0]

    ax1.plot(seq_lengths, times_no_cache, 'r-o',
             label='Without KV Cache', linewidth=2, markersize=6)
    ax1.plot(seq_lengths, times_with_cache, 'b-s',
             label='With KV Cache', linewidth=2, markersize=6)

    ax1.set_xlabel('Number of Tokens Generated', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Decoding Time: KV Cache vs No Cache', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for i, seq_len in enumerate(seq_lengths):
        speedup = times_no_cache[i] / times_with_cache[i]
        ax1.annotate(f'{speedup:.1f}x',
                     xy=(seq_len, times_with_cache[i]),
                     xytext=(seq_len, times_with_cache[i] + (times_no_cache[i] - times_with_cache[i]) * 0.4),
                     fontsize=9, color='green', fontweight='bold',
                     ha='center')

    # Right plot: Memory comparison
    ax2 = axes[1]
    ax2.plot(seq_lengths, memory_with_cache, 'b-s',
             label='KV Cache Memory', linewidth=2, markersize=6)
    ax2.fill_between(seq_lengths, 0, memory_with_cache, alpha=0.3)

    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('KV Cache Memory (MB)', fontsize=12)
    ax2.set_title('KV Cache Memory Scaling', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax2.text(seq_lengths[-1] * 0.5, memory_with_cache[-1] * 0.85,
             r'$O(N)$ memory', fontsize=12, color='blue', alpha=0.7)

    plt.tight_layout()

    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'kv_cache_benchmark.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    # ============== Summary Table ==============

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"{'Tokens':<10} {'With Cache (s)':<16} {'No Cache (s)':<16} {'Speedup':<12} {'Cache (MB)':<12}")
    print("-" * 75)

    for i, seq_len in enumerate(seq_lengths):
        speedup = times_no_cache[i] / times_with_cache[i]
        print(f"{seq_len:<10} {times_with_cache[i]:<16.4f} {times_no_cache[i]:<16.4f} {speedup:<12.1f}x {memory_with_cache[i]:<12.1f}")

    print("-" * 75)
    print(f"\nAt {seq_lengths[-1]} tokens:")
    print(f"  KV caching provides a {times_no_cache[-1] / times_with_cache[-1]:.1f}x speedup")
    print(f"  At the cost of {memory_with_cache[-1]:.1f} MB of cache memory")

    # ============== Full Model KV Cache (DeepSeek-V2) ==============

    n_layers = 60
    n_heads_ds = 128
    d_h = 128

    print("\n\nDeepSeek-V2 KV Cache (hypothetical standard MHA):")
    print(f"  Architecture: {n_layers} layers, {n_heads_ds} heads, d_h={d_h}, fp16")
    print()

    for ctx_len in [1_024, 4_096, 32_000, 131_072]:
        memory_gb = kv_cache_memory_full_model(ctx_len, n_layers, n_heads_ds, d_h)
        bytes_per_token = 2 * n_layers * n_heads_ds * d_h * 2
        print(f"  {ctx_len//1000}K context: {bytes_per_token/1e6:.1f} MB/token x {ctx_len:,} tokens = {memory_gb:.1f} GB")
