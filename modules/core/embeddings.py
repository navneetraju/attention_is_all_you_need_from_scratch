import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length: int, embedding_dim: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        # position matrix is max_seq_length * embedding_dim
        position_matrix = torch.zeros((max_seq_length, embedding_dim), dtype=torch.float32)
        for pos in range(max_seq_length):
            for i in range(embedding_dim):
                exponent = (i // 2) / embedding_dim
                if i % 2 == 0:
                    position_matrix[pos][i] = math.sin(pos / (10000 ** exponent))
                else:
                    position_matrix[pos][i] = math.cos(pos / (10000 ** exponent))
        self.register_buffer("position_matrix", position_matrix)

    def forward(self, seq_length: int):
        return self.position_matrix[:seq_length, :]
