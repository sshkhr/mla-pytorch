"""
Compare KV cache sizes across attention variants: MHA vs GQA vs MQA.

Uses direct tensor allocation (no model forward passes) to measure
cache sizes at scale, supporting long sequence lengths up to 128K.

Generates: figures/attention_variants_kv_cache.png
"""

import os
import sys

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def measure_kv_cache_size(seq_len, n_heads, d_h, n_kv_heads=None, dtype=torch.float16):
    """
    Directly allocate KV cache tensors and measure their size.
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads

    k_cache = torch.empty(1, n_kv_heads, seq_len, d_h, dtype=dtype, device=device)
    v_cache = torch.empty(1, n_kv_heads, seq_len, d_h, dtype=dtype, device=device)

    cache_bytes = k_cache.numel() * k_cache.element_size() * 2

    del k_cache, v_cache
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    return cache_bytes


if __name__ == "__main__":
    # ============== Configuration ==============

    d_model = 4096
    n_heads = 32
    n_kv_heads_gqa = 8
    d_h = d_model // n_heads

    dtype = torch.float16

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    print(f"\nConfiguration:")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads} (query heads)")
    print(f"  n_kv_heads (GQA) = {n_kv_heads_gqa}")
    print(f"  d_h = {d_h}")
    print(f"  dtype = {dtype}")
    print(f"  Sequence lengths: {seq_lengths}")
    print("=" * 60)

    # ============== Measure Cache Sizes ==============

    mha_cache_sizes = []
    gqa_cache_sizes = []
    mqa_cache_sizes = []

    print("\nMeasuring KV cache sizes (direct allocation, no attention computation)...")

    for seq_len in seq_lengths:
        mha_bytes = measure_kv_cache_size(seq_len, n_heads, d_h, n_kv_heads=n_heads, dtype=dtype)
        mha_cache_sizes.append(mha_bytes / (1024**2))

        gqa_bytes = measure_kv_cache_size(seq_len, n_heads, d_h, n_kv_heads=n_kv_heads_gqa, dtype=dtype)
        gqa_cache_sizes.append(gqa_bytes / (1024**2))

        mqa_bytes = measure_kv_cache_size(seq_len, n_heads, d_h, n_kv_heads=1, dtype=dtype)
        mqa_cache_sizes.append(mqa_bytes / (1024**2))

        print(f"  {seq_len:>6} tokens: MHA={mha_bytes/(1024**2):>8.2f} MB, "
              f"GQA={gqa_bytes/(1024**2):>7.2f} MB, MQA={mqa_bytes/(1024**2):>6.2f} MB")

    # ============== Plotting ==============

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main plot: Absolute cache sizes (log scale)
    ax.plot(np.array(seq_lengths)/1000, mha_cache_sizes, 'b-o', label='MHA (Multi-Head)', linewidth=2, markersize=6)
    ax.plot(np.array(seq_lengths)/1000, gqa_cache_sizes, 'g-s', label=f'GQA ({n_kv_heads_gqa} KV heads)', linewidth=2, markersize=6)
    ax.plot(np.array(seq_lengths)/1000, mqa_cache_sizes, 'r-^', label='MQA (1 KV head)', linewidth=2, markersize=6)

    ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
    ax.set_ylabel('KV Cache Size per layer (MB)', fontsize=12)
    ax.set_title('KV Cache Memory: MHA vs GQA vs MQA', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Add reference lines for GPU memory
    ax.axhline(y=16*1024, color='gray', linestyle='--', alpha=0.5)
    ax.text(seq_lengths[1]/1000, 18*1024, '5060 Ti 16GB', fontsize=9, color='gray')
    ax.axhline(y=80*1024, color='darkgray', linestyle='--', alpha=0.5)
    ax.text(seq_lengths[1]/1000, 90*1024, 'H100 80GB', fontsize=9, color='darkgray')

    # Inset: Memory reduction bar chart
    ax_inset = inset_axes(ax, width="25%", height="30%", loc='lower right', borderpad=2)

    gqa_reduction = (1 - n_kv_heads_gqa / n_heads) * 100
    mqa_reduction = (1 - 1 / n_heads) * 100

    bars = ax_inset.bar(['GQA', 'MQA'], [gqa_reduction, mqa_reduction],
                         color=['green', 'red'], alpha=0.7, width=0.6)

    ax_inset.set_ylabel('Reduction (%)', fontsize=9)
    ax_inset.set_title('Memory Savings vs MHA', fontsize=9)
    ax_inset.set_ylim(0, 105)
    ax_inset.tick_params(axis='both', labelsize=8)

    for bar, pct in zip(bars, [gqa_reduction, mqa_reduction]):
        ax_inset.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                      f'{pct:.0f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'attention_variants_kv_cache.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    # ============== Summary Table ==============

    print("\n" + "=" * 90)
    print("KV CACHE SIZE COMPARISON (Single Layer, fp16)")
    print("=" * 90)
    print(f"{'Seq Len':<10} {'MHA':<14} {'GQA':<14} {'MQA':<14}")
    print("-" * 90)

    for i, seq_len in enumerate(seq_lengths):
        def format_size(mb):
            if mb >= 1024:
                return f"{mb/1024:.1f} GB"
            else:
                return f"{mb:.2f} MB"

        print(f"{seq_len:<10} {format_size(mha_cache_sizes[i]):<14} {format_size(gqa_cache_sizes[i]):<14} "
              f"{format_size(mqa_cache_sizes[i]):<14}")

    print("-" * 90)
    print(f"\nMemory reduction vs MHA (constant across all sequence lengths):")
    print(f"  GQA ({n_kv_heads_gqa} KV heads): {gqa_reduction:.1f}% reduction")
    print(f"  MQA (1 KV head):    {mqa_reduction:.1f}% reduction")
