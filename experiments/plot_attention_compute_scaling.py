"""
Plot attention compute scaling: O(N^2) FLOPs for attention.

Benchmarks attention compute at GPT-2 scale on the available GPU using
F.scaled_dot_product_attention (FlashAttention backend), which performs
the same O(N^2) FLOPs without materialising the full N×N scores matrix.
This lets us measure up to 64K+ on consumer GPUs.

Estimates DeepSeek-V2 times on A100 at a fixed 70% utilisation (large
matmuls at DS-V2 scale saturate tensor cores well).

Model configs (matching the blog):
  GPT-2:        n_heads=12,  d_h=64   (d_model=768)
  DeepSeek-V2:  n_heads=128, d_h=128  (head dim independently parameterised)

Generates: figures/attention_compute_scaling.png
"""

import math
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F


# ── Model configs ────────────────────────────────────────────────────────────

MODELS = {
    "GPT-2": {"n_heads": 12, "d_h": 64},
    "DeepSeek-V2": {"n_heads": 128, "d_h": 128},
}

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# A100 SXM4: 312 TFLOP/s fp16 tensor-core peak
# At DS-V2 scale (128 heads, d_h=128) the matmuls are large enough to
# saturate tensor cores.  70% is conservative for compute-bound GEMMs.
A100_PEAK_TFLOPS = 312.0
A100_UTIL = 0.70
A100_EFFECTIVE = A100_PEAK_TFLOPS * A100_UTIL  # 218.4 TFLOP/s

# Peak fp16 tensor-core TFLOP/s for common GPUs
GPU_PEAKS = {
    "5060 Ti": 190, "5060ti": 190,
    "4090": 330, "3090": 142, "3080": 119,
    "A6000": 155, "A100": 312, "H100": 990,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def attention_flops(seq_len: int, n_heads: int, d_h: int) -> int:
    """
    FLOPs for the two attention matmuls in one layer (batch_size=1).

      QK^T:   (N, d_h) @ (d_h, N) → 2·N²·d_h  per head
      attn@V: (N, N)   @ (N, d_h) → 2·N²·d_h  per head

    Total = 4 · n_heads · N² · d_h
    """
    return 4 * n_heads * seq_len * seq_len * d_h


def format_flops(flops: float) -> str:
    if flops >= 1e15:
        return f"{flops / 1e15:.1f} PFLOP"
    if flops >= 1e12:
        return f"{flops / 1e12:.1f} TFLOP"
    if flops >= 1e9:
        return f"{flops / 1e9:.1f} GFLOP"
    return f"{flops / 1e6:.0f} MFLOP"


def format_time(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} µs"


def detect_gpu_peak() -> float:
    name = torch.cuda.get_device_name(0).lower()
    for key, val in GPU_PEAKS.items():
        if key.lower() in name:
            return val
    print(f"  ⚠ Unknown GPU; defaulting to 190 TFLOP/s peak")
    return 190.0


def short_gpu_name() -> str:
    name = torch.cuda.get_device_name(0)
    return name.replace("NVIDIA ", "").replace(" Laptop GPU", "")


# ── Benchmark ────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_attention(seq_len: int, n_heads: int, d_h: int,
                        device="cuda", dtype=torch.float16,
                        warmup=5, repeats=20):
    """
    Benchmark attention compute using F.scaled_dot_product_attention.

    This uses the FlashAttention / memory-efficient backend under the hood,
    so it performs the same O(N^2) FLOPs without materialising the full N×N
    scores matrix.  This allows benchmarking at much longer sequences.

    Returns (median_seconds, achieved_tflops) or (None, None) on OOM.
    """
    flops = attention_flops(seq_len, n_heads, d_h)

    try:
        # Shape: (batch, n_heads, seq_len, d_h)
        Q = torch.randn(1, n_heads, seq_len, d_h, device=device, dtype=dtype)
        K = torch.randn(1, n_heads, seq_len, d_h, device=device, dtype=dtype)
        V = torch.randn(1, n_heads, seq_len, d_h, device=device, dtype=dtype)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None

    def run():
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    # Warmup
    for _ in range(warmup):
        try:
            run()
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            del Q, K, V
            torch.cuda.empty_cache()
            return None, None

    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del Q, K, V
    torch.cuda.empty_cache()

    med = float(np.median(times))
    return med, flops / med / 1e12


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_label = short_gpu_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        local_peak = detect_gpu_peak()
        print(f"GPU: {gpu_label} ({gpu_mem:.1f} GB, {local_peak:.0f} TFLOP/s peak)")
        print(f"Using F.scaled_dot_product_attention (FlashAttention backend)\n")
    else:
        gpu_label, gpu_mem, local_peak = "CPU", 0, 190
        print("No CUDA GPU — skipping benchmarks.\n")

    cfg_gpt2 = MODELS["GPT-2"]
    cfg_ds = MODELS["DeepSeek-V2"]

    # ── 1.  Theoretical FLOPs ────────────────────────────────────────────────
    gpt2_flops = [attention_flops(n, **cfg_gpt2) for n in SEQ_LENGTHS]
    ds_flops   = [attention_flops(n, **cfg_ds)   for n in SEQ_LENGTHS]

    # ── 2.  Benchmark GPT-2 on local GPU ─────────────────────────────────────
    gpt2_times = []
    gpt2_achieved = []

    if device == "cuda":
        print(f"Benchmarking GPT-2 attention (12h, d_h=64) on {gpu_label} ...")
        for n in SEQ_LENGTHS:
            t, tfl = benchmark_attention(n, **cfg_gpt2, device=device)
            gpt2_times.append(t)
            gpt2_achieved.append(tfl)
            tag = f"{format_time(t)}  ({tfl:.1f} TFLOP/s)" if t else "OOM"
            print(f"  N={n:<7} {tag}")
    else:
        gpt2_times = [None] * len(SEQ_LENGTHS)
        gpt2_achieved = [None] * len(SEQ_LENGTHS)

    # ── 3.  DS-V2 theoretical times on A100 ──────────────────────────────────
    print(f"\nDeepSeek-V2 A100 estimate: {A100_PEAK_TFLOPS:.0f} peak × "
          f"{A100_UTIL*100:.0f}% util = {A100_EFFECTIVE:.0f} effective TFLOP/s")

    ds_times_est = [f / (A100_EFFECTIVE * 1e12) for f in ds_flops]

    # GPT-2 OOM extrapolation (if any)
    valid = [t for t in gpt2_achieved if t is not None]
    gpt2_peak = max(valid) if valid else local_peak * 0.05
    gpt2_oom = [t is None for t in gpt2_times]

    gpt2_time_col = [
        gpt2_times[i] if not gpt2_oom[i]
        else gpt2_flops[i] / (gpt2_peak * 1e12)
        for i in range(len(SEQ_LENGTHS))
    ]

    # ── 4.  Print table ──────────────────────────────────────────────────────
    W = [10, 16, 18, 18, 20]
    div = "=" * (sum(W) + 8)

    print(f"\n{div}")
    print(f"{'Seq Len':>{W[0]}}  {'GPT-2':>{W[1]}}  {'GPT-2':>{W[2]}}  "
          f"{'DeepSeek-V2':>{W[3]}}  {'DeepSeek-V2':>{W[4]}}")
    print(f"{'':>{W[0]}}  {'FLOPs/layer':>{W[1]}}  "
          f"{'Time (' + gpu_label + ')':>{W[2]}}  "
          f"{'FLOPs/layer':>{W[3]}}  {'Time (A100 est.)':>{W[4]}}")
    print(div)

    for i, n in enumerate(SEQ_LENGTHS):
        fg = format_flops(gpt2_flops[i])
        fd = format_flops(ds_flops[i])

        tg = format_time(gpt2_time_col[i])
        if gpt2_oom[i]:
            tg = f"({tg})*"

        # All DS-V2 times are theoretical estimates
        td = format_time(ds_times_est[i])

        print(f"{n:>{W[0]}}  {fg:>{W[1]}}  {tg:>{W[2]}}  "
              f"{fd:>{W[3]}}  {td:>{W[4]}}")

    print(div)
    if any(gpt2_oom):
        print(f"* GPT-2 OOM rows extrapolated at {gpt2_peak:.0f} TFLOP/s "
              f"(measured peak on {gpu_label}).")
    print(f"  DeepSeek-V2 estimated at {A100_EFFECTIVE:.0f} TFLOP/s "
          f"(A100 {A100_PEAK_TFLOPS:.0f} peak × {A100_UTIL*100:.0f}% util).\n")

    # ── 5.  Plot ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    S = np.array(SEQ_LENGTHS, dtype=float)

    # ── Left panel: FLOPs ────────────────────────────────────────────────────
    ax1.loglog(S, gpt2_flops, "b-o", lw=2, ms=5, label="GPT-2 (12h, d_h=64)")
    ax1.loglog(S, ds_flops,   "r-s", lw=2, ms=5, label="DeepSeek-V2 (128h, d_h=128)")

    # O(N²) reference
    rx = np.array([S[0], S[-1]])
    ry0 = gpt2_flops[0] * 0.3
    ax1.loglog(rx, ry0 * (rx / rx[0])**2, "k:", alpha=0.4, lw=1.5)
    ax1.text(rx[0] * 3, ry0 * 0.5, r"$\propto N^2$", fontsize=12, alpha=0.6)

    ax1.set_xlabel("Sequence Length", fontsize=12)
    ax1.set_ylabel("FLOPs (per layer)", fontsize=12)
    ax1.set_title("Attention FLOPs vs Sequence Length", fontsize=13)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_flops(x)))

    # ── Right panel: wall time ───────────────────────────────────────────────
    gpt2_ms = np.array([t * 1e3 for t in gpt2_time_col])
    ds_ms   = np.array([t * 1e3 for t in ds_times_est])

    # GPT-2: measured (solid) + OOM extension (dashed)
    m = np.array([not o for o in gpt2_oom])
    if m.any():
        ax2.loglog(S[m], gpt2_ms[m], "b-o", lw=2, ms=6,
                   label=f"GPT-2 measured ({gpu_label})")
    if any(gpt2_oom) and m.any():
        last_m = int(np.where(m)[0][-1])
        ext_idx = [last_m] + [j for j, o in enumerate(gpt2_oom) if o]
        ax2.loglog(S[ext_idx], gpt2_ms[ext_idx],
                   "b--^", lw=1.5, ms=5, alpha=0.6,
                   label="GPT-2 theoretical (OOM)")

    # DS-V2: all theoretical
    ax2.loglog(S, ds_ms, "r-s", lw=2, ms=5,
               label=f"DeepSeek-V2 estimated (A100, {A100_UTIL*100:.0f}% util)")

    ax2.set_xlabel("Sequence Length", fontsize=12)
    ax2.set_ylabel("Wall Time (ms)", fontsize=12)
    ax2.set_title("Attention Time vs Sequence Length (per layer)", fontsize=13)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "attention_compute_scaling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Figure saved to {out}")
    plt.show()