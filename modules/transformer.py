import torch.nn as nn

from modules.blocks.decoder_block import TransformerDecoderBlock
from modules.blocks.encoder_block import TransformerEncoderBlock
from modules.core.embeddings import TokenEmbedding, PositionalEmbedding


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
        self.encoder_positional_embedding = PositionalEmbedding(max_seq_length=max_seq_length,
                                                                embedding_dim=embedding_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, feed_forward_dim=feed_forward_dim)
            for _ in range(num_encoder_layers)
        ])

        ##### DECODER #####
        self.decoder_token_embedding = TokenEmbedding(vocab_size=tgt_vocab_size, embedding_dim=embedding_dim)
        self.decoder_positional_embedding = PositionalEmbedding(max_seq_length=max_seq_length,
                                                                embedding_dim=embedding_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, feed_forward_dim=feed_forward_dim)
            for _ in range(num_decoder_layers)
        ])

        ##### FFN Layer #####
        self.linear = nn.Linear(embedding_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        # x is of shape (batch_size, seq_length)

        ##### ENCODER #####
        input_embedding = self.encoder_token_embedding(src) + self.encoder_positional_embedding(src.shape[1]).unsqueeze(
            0)
        for encoder in self.encoder_blocks:
            input_embedding = encoder(input_embedding)

        ##### DECODER #####
        decoder_embedding = self.decoder_token_embedding(tgt) + self.decoder_positional_embedding(
            tgt.shape[1]).unsqueeze(0)

        for decoder in self.decoder_blocks:
            decoder_embedding = decoder(decoder_embedding, input_embedding, input_embedding)

        return self.linear(decoder_embedding)
