import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - Naive implementation.

    Compresses KV representations into a low-dimensional latent space.
    During decoding, only the compressed latent is cached.
    Expands latent to full K, V before attention computation.

    KV cache size per token: d_c (vs 2 * n_heads * d_h for standard MHA)
    """
    def __init__(self, d_model, n_heads, d_c):
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.d_c = d_c

        self.W_q = nn.Linear(d_model, n_heads * self.d_h, bias=False)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        self.W_uk = nn.Linear(d_c, n_heads * self.d_h, bias=False)
        self.W_uv = nn.Linear(d_c, n_heads * self.d_h, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_h, d_model, bias=False)

        self.latent_cache = None

    def forward(self, x, use_cache=False):
        B, N, _ = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_h).transpose(1, 2)

        L_kv = self.W_dkv(x)

        if use_cache:
            if self.latent_cache is not None:
                L_kv = torch.cat([self.latent_cache, L_kv], dim=1)
            self.latent_cache = L_kv

        # Expand latent to full K, V (on-the-fly, not cached)
        K = self.W_uk(L_kv).view(B, -1, self.n_heads, self.d_h).transpose(1, 2)
        V = self.W_uv(L_kv).view(B, -1, self.n_heads, self.d_h).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(out)

    def reset_cache(self):
        self.latent_cache = None

    def cache_size_bytes(self):
        if self.latent_cache is None:
            return 0
        return self.latent_cache.numel() * self.latent_cache.element_size()

    def cache_size_per_token(self):
        return self.d_c


class MultiHeadLatentAttentionAbsorbed(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - With weight absorption.

    Absorbs the KV up-projections into Q and O matrices so that
    attention is computed directly in the latent space, without
    ever expanding to full K, V dimensions.

    W_q' = W_q @ W_uk^T  (queries project to latent-compatible space)
    W_o' = W_uv @ W_o    (output combines value up-projection and output projection)

    KV cache size per token: d_c
    """
    def __init__(self, d_model, n_heads, d_c):
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.d_c = d_c

        # ABSORBED weights: Q projects to latent space (per head)
        # W_q' = W_q @ W_uk^T : (d_model, n_heads*d_h) @ (n_heads*d_h, d_c) -> (d_model, n_heads*d_c)
        self.W_q_absorbed = nn.Linear(d_model, n_heads * d_c, bias=False)

        # KV down-projection (unchanged)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)

        # ABSORBED output: combines value up-projection and output projection
        self.W_o_absorbed = nn.Linear(n_heads * d_c, d_model, bias=False)

        self.latent_cache = None

    def forward(self, x, use_cache=False):
        B, N, _ = x.shape

        # Query projected to latent-compatible space (per head)
        # Shape: (B, N, n_heads, d_c) - note d_c instead of d_h!
        Q = self.W_q_absorbed(x).view(B, N, self.n_heads, self.d_c).transpose(1, 2)

        L_kv = self.W_dkv(x)  # (B, N, d_c)

        if use_cache:
            if self.latent_cache is not None:
                L_kv = torch.cat([self.latent_cache, L_kv], dim=1)
            self.latent_cache = L_kv

        # NO EXPANSION NEEDED!
        # L_kv serves as both K and V in latent space
        # Broadcast across heads: (B, 1, N, d_c) -> (B, n_heads, N, d_c)
        L_kv_expanded = L_kv.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # Attention directly in latent space
        scores = torch.matmul(Q, L_kv_expanded.transpose(-2, -1)) / math.sqrt(self.d_c)
        attn = F.softmax(scores, dim=-1)

        # Output is attention-weighted latents (not full V!)
        out = torch.matmul(attn, L_kv_expanded)  # (B, n_heads, N, d_c)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o_absorbed(out)

    def reset_cache(self):
        self.latent_cache = None

    def cache_size_bytes(self):
        if self.latent_cache is None:
            return 0
        return self.latent_cache.numel() * self.latent_cache.element_size()

    def cache_size_per_token(self):
        return self.d_c
