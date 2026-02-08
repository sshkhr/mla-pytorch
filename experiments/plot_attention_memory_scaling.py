"""
Plot attention memory scaling: O(N^2) attention matrix vs O(N) QKV tensors.

Compares GPT-2 scale (768 dim, 12 heads) vs DeepSeek-V2 scale (5120 dim, 128 heads).
Generates: figures/attention_memory_scaling.png
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def attention_tensor_sizes(batch_size, seq_len, d_model, n_heads, dtype=torch.float16):
    """Calculate tensor sizes for MHA computation."""
    d_h = d_model // n_heads
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    # Per-head shapes
    Q_shape = (batch_size, n_heads, seq_len, d_h)
    K_shape = (batch_size, n_heads, seq_len, d_h)
    V_shape = (batch_size, n_heads, seq_len, d_h)
    attn_scores_shape = (batch_size, n_heads, seq_len, seq_len)  # The N×N matrix

    # Memory in bytes
    qkv_memory = 3 * batch_size * n_heads * seq_len * d_h * bytes_per_elem
    attn_matrix_memory = batch_size * n_heads * seq_len * seq_len * bytes_per_elem

    return {
        "Q/K/V each": Q_shape,
        "Attention scores": attn_scores_shape,
        "QKV memory (MB)": qkv_memory / (1024**2),
        "Attention matrix memory (MB)": attn_matrix_memory / (1024**2),
    }


def attention_memory_stats(seq_len, d_model, n_heads, dtype=torch.float16):
    """Calculate memory for QKV and attention matrix."""
    d_h = d_model // n_heads
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    # Memory in bytes (batch_size=1)
    qkv_memory = 3 * n_heads * seq_len * d_h * bytes_per_elem
    attn_matrix_memory = n_heads * seq_len * seq_len * bytes_per_elem

    return {
        "qkv_gb": qkv_memory / (1024**3),
        "attn_gb": attn_matrix_memory / (1024**3),
    }


if __name__ == "__main__":
    # Print key numbers
    print("GPT-2 scale (d=768, 12 heads):")
    for seq_len in [1024, 4096, 16384, 65536]:
        stats = attention_tensor_sizes(1, seq_len, 768, 12)
        print(f"  seq_len={seq_len:>5}: attn matrix = {stats['Attention matrix memory (MB)']:>8.1f} MB")

    print("\nDeepSeek-V2 scale (d=5120, 128 heads):")
    for seq_len in [1024, 4096, 16384, 65536]:
        stats = attention_tensor_sizes(1, seq_len, 5120, 128)
        print(f"  seq_len={seq_len:>5}: attn matrix = {stats['Attention matrix memory (MB)']:>8.1f} MB")

    # Sequence lengths to plot
    seq_lengths = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])

    # Calculate memory for both scales (in GB)
    gpt2_qkv, gpt2_attn = [], []
    ds_qkv, ds_attn = [], []

    for seq_len in seq_lengths:
        stats = attention_memory_stats(seq_len, 768, 12)
        gpt2_qkv.append(stats["qkv_gb"])
        gpt2_attn.append(stats["attn_gb"])

        stats = attention_memory_stats(seq_len, 5120, 128)
        ds_qkv.append(stats["qkv_gb"])
        ds_attn.append(stats["attn_gb"])

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Linear scale (shows the parabolic quadratic blowup)
    ax1 = axes[0]
    ax1.plot(seq_lengths/1000, gpt2_qkv, 'b--', label='GPT-2: Q,K,V tensors', linewidth=2, marker='o', markersize=4)
    ax1.plot(seq_lengths/1000, gpt2_attn, 'b-', label='GPT-2: Attention matrix', linewidth=2, marker='s', markersize=4)
    ax1.plot(seq_lengths/1000, ds_qkv, 'r--', label='DeepSeek-V2: Q,K,V tensors', linewidth=2, marker='o', markersize=4)
    ax1.plot(seq_lengths/1000, ds_attn, 'r-', label='DeepSeek-V2: Attention matrix', linewidth=2, marker='s', markersize=4)

    ax1.axhline(y=80, color='black', linestyle='-', alpha=0.7, label='H100 80GB')
    ax1.set_xlabel('Sequence Length (K tokens)', fontsize=12)
    ax1.set_ylabel('Memory (GB)', fontsize=12)
    ax1.set_title('Attention Memory Scaling (Linear Scale)', fontsize=13)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 70)

    # Right plot: Log-log scale (shows O(N) vs O(N²) as slopes)
    ax2 = axes[1]
    ax2.loglog(seq_lengths, gpt2_qkv, 'b--', label='GPT-2: Q,K,V (slope=1)', linewidth=2, marker='o', markersize=4)
    ax2.loglog(seq_lengths, gpt2_attn, 'b-', label='GPT-2: Attn matrix (slope=2)', linewidth=2, marker='s', markersize=4)
    ax2.loglog(seq_lengths, ds_qkv, 'r--', label='DeepSeek-V2: Q,K,V (slope=1)', linewidth=2, marker='o', markersize=4)
    ax2.loglog(seq_lengths, ds_attn, 'r-', label='DeepSeek-V2: Attn matrix (slope=2)', linewidth=2, marker='s', markersize=4)

    # Reference lines for O(N) and O(N²)
    ref_x = np.array([512, 65536])
    ax2.loglog(ref_x, ref_x/50/1024, 'k:', alpha=0.4, linewidth=1.5)
    ax2.loglog(ref_x, (ref_x**2)/1e5/1024, 'k:', alpha=0.4, linewidth=1.5)
    ax2.text(1200, 0.015, r'$O(N)$', fontsize=11, alpha=0.6)
    ax2.text(1200, 0.8, r'$O(N^2)$', fontsize=11, alpha=0.6)

    ax2.axhline(y=80, color='black', linestyle='-', alpha=0.7, label='H100 80GB')
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Memory (GB)', fontsize=12)
    ax2.set_title('Log-Log Scale (slope = complexity exponent)', fontsize=13)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'attention_memory_scaling.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    # Print key data points
    print("\nKey data points:")
    print(f"{'Seq Len':<10} {'GPT-2 Attn':<15} {'DeepSeek-V2 Attn':<15}")
    print("-" * 45)
    for i, seq_len in enumerate(seq_lengths):
        gpt2_str = f"{gpt2_attn[i]*1024:,.0f} MB" if gpt2_attn[i] < 1 else f"{gpt2_attn[i]:,.1f} GB"
        ds_str = f"{ds_attn[i]*1024:,.0f} MB" if ds_attn[i] < 1 else f"{ds_attn[i]:,.1f} GB"
        print(f"{seq_len:<10} {gpt2_str:<15} {ds_str:<15}")
