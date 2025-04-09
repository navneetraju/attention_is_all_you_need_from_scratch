import torch.nn as nn
from torch.nn import Embedding
import torch
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

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
                    position_matrix[pos][i] = torch.sin(pos / torch.pow(10000, exponent))
                else:
                    position_matrix[pos][i] = torch.cos(pos / torch.pow(10000, exponent))
        self.register_buffer("position_matrix", position_matrix)

    def forward(self, seq_length: int):
        return self.position_matrix[:seq_length, :]

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
        # query, key, value are (batch_size, seq_length, embedding_dim)
        batch_size, seq_length, embedding_dim = query.shape
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        Q = Q.view(batch_size, self.num_heads, seq_length, self.h)
        K = K.view(batch_size, self.num_heads, seq_length, self.h)
        V = V.view(batch_size, self.num_heads, seq_length, self.h)

        def scaled_dot_product_attention(q, k, v):
            scores = q @ k.transpose(-2, -1)
            scaled_scores = scores / (self.h ** 0.5)

            if self.mask:
                mask = torch.triu(torch.ones(seq_length, seq_length, device=q.device), diagonal=1).bool()
                scaled_scores = torch.masked_fill(scaled_scores, mask, -torch.inf)

            attention_weights = F.softmax(scaled_scores, dim=-1)
            return attention_weights @ v

        head_outputs = []
        for i in range(self.num_heads):
            Q_i = Q[:, i, :, :]
            K_i = K[:, i, :, :]
            V_i = V[:, i, :, :]
            head_output = scaled_dot_product_attention(Q_i, K_i, V_i)
            head_outputs.append(head_output)

        head_outputs_concat = torch.cat(head_outputs, dim=-1)
        out = self.ffn(head_outputs_concat)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim

        self.multi_headed_attention = MultiHeadedAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.layer_norm_attn = nn.LayerNorm(embedding_dim)
        self.ffn_1 = nn.Linear(embedding_dim, feed_forward_dim)
        self.ffn_2 = nn.Linear(feed_forward_dim, embedding_dim)
        self.layer_norm_ffn = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # MHA first
        mha_out = self.multi_headed_attention(x, x, x)
        # Add and norm
        attn_out = self.layer_norm_attn(mha_out + x)

        # FFN layers
        out = F.relu(self.ffn_1(attn_out))
        out = self.ffn_2(out)
        # add and norm
        ffn_out = self.layer_norm_ffn(attn_out + out)

        return ffn_out

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim

        self.masked_multi_headed_attention = MultiHeadedAttention(embedding_dim=embedding_dim, num_heads=num_heads,
                                                                  mask=True)
        self.layer_norm_masked_attn = nn.LayerNorm(embedding_dim)
        self.cross_multi_headed_attention = MultiHeadedAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.layer_norm_cross_attn = nn.LayerNorm(embedding_dim)
        self.ffn_1 = nn.Linear(embedding_dim, feed_forward_dim)
        self.ffn_2 = nn.Linear(feed_forward_dim, embedding_dim)
        self.layer_norm_ffn = nn.LayerNorm(embedding_dim)

    def forward(self, x, key, value):
        # Masked MHA
        masked_mha_out = self.masked_multi_headed_attention(x, x, x)
        # add and norm
        query = self.layer_norm_masked_attn(masked_mha_out + x)

        # Cross attention
        cross_mha_out = self.cross_multi_headed_attention(query, key, value)
        # add and norm
        attn_out = self.layer_norm_cross_attn(cross_mha_out + query)

        # FFN Layers
        out = F.relu(self.ffn_1(attn_out))
        out = self.ffn_2(out)
        # add and norm
        out = self.layer_norm_ffn(attn_out + out)

        return out


class TransformerEncoderDecoder(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            max_seq_length: int,
            embedding_dim: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            num_heads: int,
            feed_forward_dim: int
    ):
        super().__init__()

        ##### ENCODER #####
        self.encoder_token_embedding = TokenEmbedding(vocab_size=src_vocab_size, embedding_dim=embedding_dim)
        self.encoder_positional_embedding = PositionalEmbedding(max_seq_length=max_seq_length, embedding_dim=embedding_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, feed_forward_dim=feed_forward_dim)
            for _ in range(num_encoder_layers)
        ])

        ##### DECODER #####
        self.decoder_token_embedding = TokenEmbedding(vocab_size=tgt_vocab_size, embedding_dim=embedding_dim)
        self.decoder_positional_embedding = PositionalEmbedding(max_seq_length=max_seq_length, embedding_dim=embedding_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, feed_forward_dim=feed_forward_dim)
            for _ in range(num_decoder_layers)
        ])

        ##### FFN Layer #####
        self.linear = nn.Linear(embedding_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        # x is of shape (batch_size, seq_length)

        ##### ENCODER #####
        input_embedding = self.encoder_token_embedding(src) + self.encoder_positional_embedding(src.shape[1])
        for encoder in self.encoder_blocks:
            input_embedding = encoder(input_embedding)

        ##### DECODER #####
        decoder_embedding = self.decoder_token_embedding(tgt) + self.decoder_positional_embedding(tgt.shape[1])

        for decoder in self.decoder_blocks:
            decoder_embedding = decoder(decoder_embedding, input_embedding, input_embedding)

        return self.linear(decoder_embedding)