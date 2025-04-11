import torch.nn as nn
import torch.nn.functional as F

from modules.core.multi_head_attention import MultiHeadedAttention


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
