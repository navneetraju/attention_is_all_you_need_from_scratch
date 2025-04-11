import torch.nn as nn
import torch.nn.functional as F

from modules.core.multi_head_attention import MultiHeadedAttention


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
