import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention with optional KV Cache.
    Each head has its own K and V projections.

    KV cache size per token: 2 * n_heads * d_h
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=False):
        B, seq_len, _ = x.shape

        Q = self.W_q(x).view(B, seq_len, self.n_heads, self.d_h).transpose(1, 2)
        K = self.W_k(x).view(B, seq_len, self.n_heads, self.d_h).transpose(1, 2)
        V = self.W_v(x).view(B, seq_len, self.n_heads, self.d_h).transpose(1, 2)

        if use_cache:
            if self.k_cache is None:
                self.k_cache = K
                self.v_cache = V
            else:
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)
            K, V = self.k_cache, self.v_cache

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        return self.W_o(output)

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    def cache_size_bytes(self):
        if self.k_cache is None:
            return 0
        return self.k_cache.numel() * self.k_cache.element_size() * 2
