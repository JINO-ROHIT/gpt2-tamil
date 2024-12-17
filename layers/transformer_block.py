import torch
import torch.nn as nn

from layers import FeedForward, MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension: int, context_length: int, num_heads: int, bias: bool, dropout: float, scaling_factor: int, *kwargs):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.context_length = context_length
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout
        self.scaling_factor = scaling_factor

        self.mha = MultiHeadAttention(self.embedding_dimension, self.embedding_dimension, self.num_heads, self.context_length)
        self.ff = FeedForward(self.embedding_dimension, self.scaling_factor, self.bias)
        self.norm1 = nn.LayerNorm(self.embedding_dimension, eps = 1e-5)
        self.norm2 = nn.LayerNorm(self.embedding_dimension, eps = 1e-5)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x

if __name__ == '__main__':
    batch_size = 8
    context_len = 256
    embed_dim = 768
    num_heads = 12
    bias = False
    dropout = 0.0
    scaling_factor = 4

    embeddings = torch.randn((batch_size, context_len, embed_dim), device = 'cuda' if torch.cuda.is_available else 'cpu')

    transformer_block = TransformerBlock(embed_dim, context_len, num_heads, bias, dropout, scaling_factor).to(device = 'cuda' if torch.cuda.is_available else 'cpu')

    print(transformer_block(embeddings), transformer_block(embeddings).shape)
