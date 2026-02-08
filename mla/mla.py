import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - Naive implementation.

    Compresses both Q and KV representations into low-dimensional latent spaces.
    During decoding, only the compressed KV latent is cached.
    Expands latent to full K, V before attention computation.

    Query compression: x -> W_dq -> RMSNorm -> W_uq -> Q  (not cached)
    KV compression:    x -> W_dkv -> L_kv (cached) -> W_uk/W_uv -> K, V

    KV cache size per token: d_c (vs 2 * n_heads * d_h for standard MHA)
    """
    def __init__(self, d_model, n_heads, d_c, d_cq, max_seq_len=4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.d_c = d_c
        self.d_cq = d_cq
        self.max_seq_len = max_seq_len

        # Query compression: down-project, normalize, up-project
        self.W_dq = nn.Linear(d_model, d_cq, bias=False)
        self.q_norm = RMSNorm(d_cq)
        self.W_uq = nn.Linear(d_cq, n_heads * self.d_h, bias=False)

        # KV compression
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        self.W_uk = nn.Linear(d_c, n_heads * self.d_h, bias=False)
        self.W_uv = nn.Linear(d_c, n_heads * self.d_h, bias=False)

        self.W_o = nn.Linear(n_heads * self.d_h, d_model, bias=False)

        # Pre-allocated latent cache
        self.register_buffer('latent_cache', torch.zeros(1, max_seq_len, d_c))
        self.cache_position = 0

    def forward(self, x, use_cache=False):
        B, N, _ = x.shape

        # Query: down-project -> normalize -> up-project
        L_q = self.q_norm(self.W_dq(x))
        Q = self.W_uq(L_q).view(B, N, self.n_heads, self.d_h).transpose(1, 2)

        # KV: compress to latent
        L_kv = self.W_dkv(x)

        if use_cache:
            start = self.cache_position
            end = start + N
            self.latent_cache[:B, start:end, :] = L_kv
            self.cache_position = end
            L_kv = self.latent_cache[:B, :end, :]

        # Expand latent to full K, V (on-the-fly, not cached)
        K = self.W_uk(L_kv).view(B, -1, self.n_heads, self.d_h).transpose(1, 2)
        V = self.W_uv(L_kv).view(B, -1, self.n_heads, self.d_h).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(out)

    def reset_cache(self):
        self.latent_cache.zero_()
        self.cache_position = 0

    def cache_size_bytes(self):
        if self.cache_position == 0:
            return 0
        return self.cache_position * self.d_c * self.latent_cache.element_size()

    def cache_size_per_token(self):
        return self.d_c


class MultiHeadLatentAttentionAbsorbed(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - With weight absorption.

    Absorbs the KV up-projections into Q and O matrices so that
    attention is computed directly in the latent space, without
    ever expanding to full K, V dimensions.

    Query path:  x -> W_dq -> RMSNorm -> W_q_absorbed -> Q  (in latent space)
    Where W_q_absorbed = W_uq @ W_uk^T  (absorbed)

    Output path: attn @ L_kv -> W_o_absorbed
    Where W_o_absorbed = W_uv @ W_o  (absorbed)

    KV cache size per token: d_c
    """
    def __init__(self, d_model, n_heads, d_c, d_cq, max_seq_len=4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.d_c = d_c
        self.d_cq = d_cq
        self.max_seq_len = max_seq_len

        # Query compression: down-project, normalize
        self.W_dq = nn.Linear(d_model, d_cq, bias=False)
        self.q_norm = RMSNorm(d_cq)

        # ABSORBED query: W_q' = W_uq @ W_uk^T, applied after query compression
        # Maps from d_cq to n_heads * d_c (query in latent-compatible space per head)
        self.W_q_absorbed = nn.Linear(d_cq, n_heads * d_c, bias=False)

        # KV down-projection (unchanged)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)

        # ABSORBED output: combines value up-projection and output projection
        self.W_o_absorbed = nn.Linear(n_heads * d_c, d_model, bias=False)

        # Pre-allocated latent cache
        self.register_buffer('latent_cache', torch.zeros(1, max_seq_len, d_c))
        self.cache_position = 0

    def forward(self, x, use_cache=False):
        B, N, _ = x.shape

        # Query: down-project -> normalize -> absorbed projection to latent space
        L_q = self.q_norm(self.W_dq(x))
        # Shape: (B, N, n_heads, d_c) - note d_c instead of d_h!
        Q = self.W_q_absorbed(L_q).view(B, N, self.n_heads, self.d_c).transpose(1, 2)

        L_kv = self.W_dkv(x)  # (B, N, d_c)

        if use_cache:
            start = self.cache_position
            end = start + N
            self.latent_cache[:B, start:end, :] = L_kv
            self.cache_position = end
            L_kv = self.latent_cache[:B, :end, :]

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
        self.latent_cache.zero_()
        self.cache_position = 0

    def cache_size_bytes(self):
        if self.cache_position == 0:
            return 0
        return self.cache_position * self.d_c * self.latent_cache.element_size()

    def cache_size_per_token(self):
        return self.d_c
