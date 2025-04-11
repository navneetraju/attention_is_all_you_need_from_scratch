import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mask: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.h = embedding_dim // num_heads
        self.mask = mask

        # Projection matrices
        self.W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.ffn = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value):
        batch_size, seq_length_q, embed_dim_q = query.shape
        _, seq_length_k, embed_dim_k = key.shape

        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        Q = Q.view(batch_size, self.num_heads, seq_length_q, self.h)
        K = K.view(batch_size, self.num_heads, seq_length_k, self.h)
        V = V.view(batch_size, self.num_heads, seq_length_k, self.h)

        def scaled_dot_product_attention(q, k, v):
            scores = q @ k.transpose(-2, -1)
            scaled_scores = scores / (self.h ** 0.5)
            if self.mask:
                mask = torch.triu(torch.ones(seq_length_q, seq_length_q, device=q.device), diagonal=1).bool()
                # Note: When masking in cross-attention, you might need to adjust the mask shape to fit (seq_length_q, seq_length_k)
                scaled_scores = torch.masked_fill(scaled_scores, mask, -torch.inf)
            attention_weights = F.softmax(scaled_scores, dim=-1)
            return attention_weights @ v

        head_outputs = []
        for i in range(self.num_heads):
            Q_i = Q[:, i, :, :]  # shape: (batch_size, seq_length_q, head_dim)
            K_i = K[:, i, :, :]  # shape: (batch_size, seq_length_k, head_dim)
            V_i = V[:, i, :, :]  # shape: (batch_size, seq_length_k, head_dim)
            head_output = scaled_dot_product_attention(Q_i, K_i, V_i)
            head_outputs.append(head_output)

        head_outputs_concat = torch.cat(head_outputs, dim=-1)
        out = self.ffn(head_outputs_concat)
        return out

