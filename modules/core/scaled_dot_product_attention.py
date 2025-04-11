import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_dim: int, mask: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mask = mask
        self.W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        # x will have shape (batch_size, seq_length, embedding_dim)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1)
        scaled_scores = scores / (self.embedding_dim ** 0.5)
        seq_length = x.shape[1]
        if self.mask:
            mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()
            scaled_scores = torch.masked_fill(scaled_scores, mask, -torch.inf)

        attention_weights = F.softmax(scaled_scores, dim=-1)  # apply to the last dimension (i.e row wise)

        out = attention_weights @ V
        return out
