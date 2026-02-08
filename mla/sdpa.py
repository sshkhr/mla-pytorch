import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as described in "Attention Is All You Need"
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query tensor of shape (batch_size, seq_len_q, d_k)
            K: Key tensor of shape (batch_size, seq_len_k, d_k)
            V: Value tensor of shape (batch_size, seq_len_v, d_v)
            mask: Optional mask tensor of shape (batch_size, seq_len_q, seq_len_k)

        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_v)
            attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, d_k = Q.shape
        seq_len_k = K.shape[1]

        # Step 1: Compute raw attention scores Q @ K^T
        # (batch_size, seq_len_q, d_k) @ (batch_size, d_k, seq_len_k) -> (batch_size, seq_len_q, seq_len_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Step 2: Scale by sqrt(d_k)
        attention_scores = attention_scores / math.sqrt(d_k)

        # Step 3: Apply mask (if provided)
        if mask is not None:
            attention_scores = attention_scores + mask  # mask should contain -inf for masked positions

        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len_q, seq_len_k)

        # Step 5: Apply attention weights to values
        # (batch_size, seq_len_q, seq_len_k) @ (batch_size, seq_len_k, d_v) -> (batch_size, seq_len_q, d_v)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
