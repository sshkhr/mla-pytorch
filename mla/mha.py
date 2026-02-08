import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention with optional KV Cache.
    Each head has its own K and V projections.

    Uses pre-allocated cache buffers for efficient autoregressive decoding
    (no memory allocation during decode steps).

    KV cache size per token: 2 * n_heads * d_h
    """
    def __init__(self, d_model, n_heads, max_seq_len=4096):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Pre-allocated KV cache buffers
        self.register_buffer('k_cache', torch.zeros(1, n_heads, max_seq_len, self.d_h))
        self.register_buffer('v_cache', torch.zeros(1, n_heads, max_seq_len, self.d_h))
        self.cache_position = 0

    def forward(self, x, use_cache=False):
        B, seq_len, _ = x.shape

        Q = self.W_q(x).view(B, seq_len, self.n_heads, self.d_h).transpose(1, 2)
        K = self.W_k(x).view(B, seq_len, self.n_heads, self.d_h).transpose(1, 2)
        V = self.W_v(x).view(B, seq_len, self.n_heads, self.d_h).transpose(1, 2)

        if use_cache:
            start = self.cache_position
            end = start + seq_len
            self.k_cache[:B, :, start:end, :] = K
            self.v_cache[:B, :, start:end, :] = V
            self.cache_position = end
            K = self.k_cache[:B, :, :end, :]
            V = self.v_cache[:B, :, :end, :]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        return self.W_o(output)

    def reset_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_position = 0

    def cache_size_bytes(self):
        if self.cache_position == 0:
            return 0
        return 2 * self.cache_position * self.n_heads * self.d_h * self.k_cache.element_size()
