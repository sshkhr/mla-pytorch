# Multi-Head Latent Attention (MLA) in PyTorch

Companion code for the blog post [Understanding Multi-Head Latent Attention](https://shashankshekhar.com/blog/flashmla/flashmla-1-mla), part of a series on [DeepSeek's FlashMLA](https://github.com/deepseek-ai/FlashMLA).

## Attention Implementations

Clean, minimal PyTorch implementations of attention mechanisms discussed in the blog:

| Module | Class | Description |
|--------|-------|-------------|
| `mla/sdpa.py` | `ScaledDotProductAttention` | Scaled dot-product attention ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) |
| `mla/mha.py` | `MultiHeadAttention` | Standard multi-head attention with optional KV cache |
| `mla/gqa.py` | `GroupedQueryAttention` | Grouped-query attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) |
| `mla/mqa.py` | `MultiQueryAttention` | Multi-query attention ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) |
| `mla/mla.py` | `MultiHeadLatentAttention` | MLA with low-rank KV compression ([DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434)) |
| `mla/mla.py` | `MultiHeadLatentAttentionAbsorbed` | MLA with weight absorption trick |

## Experiments

Scripts that reproduce the figures from the blog post:

| Script | Figure | Description |
|--------|--------|-------------|
| `experiments/plot_attention_memory_scaling.py` | `attention_memory_scaling.png` | O(N^2) attention matrix memory scaling |
| `experiments/plot_attention_compute_scaling.py` | `attention_compute_scaling.png` | O(N^2) attention FLOPs and wall-clock time scaling |
| `experiments/plot_kv_cache_benchmark.py` | `kv_cache_benchmark.png` | Decoding speedup with KV caching |
| `experiments/plot_attention_variants_kv_cache.py` | `attention_variants_kv_cache.png` | KV cache size: MHA vs GQA vs MQA |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from mla import MultiHeadAttention, MultiHeadLatentAttention

# Standard MHA
mha = MultiHeadAttention(d_model=512, n_heads=8)
output = mha(x)  # x: (batch, seq_len, 512)

# MLA with latent compression (KV + query compression)
mla = MultiHeadLatentAttention(d_model=512, n_heads=8, d_c=64, d_cq=96)
output = mla(x, use_cache=True)  # caches compressed latent
```

Run experiments:

```bash
python experiments/plot_attention_memory_scaling.py
python experiments/plot_attention_compute_scaling.py
python experiments/plot_kv_cache_benchmark.py
python experiments/plot_attention_variants_kv_cache.py
```

Figures are saved to `figures/`.
