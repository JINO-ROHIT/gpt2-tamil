import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dimension: int , output_dimension: int, num_heads: int = 12, context_length: int = 256, dropout: float = 0, bias: bool = False):
        super().__init__()

        assert output_dimension % num_heads == 0, "head dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dimension = output_dimension // num_heads

        self.qkv = nn.Linear(input_dimension, 3 * output_dimension, bias = bias)
        self.proj = nn.Linear(output_dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_length, embedding_dimension = x.shape

        qkv = self.qkv(x) # (bs, seq_len, embed_dim) --> (bs, seq_len, 3 * embed_dim)
        qkv = qkv.view(bs, seq_length, 3, self.num_heads, self.head_dimension) # (bs, seq_len, 3 * embed_dim) --> (bs, seq_len, 3, num_heads, head_dim)
        # (bs, seq_len, 3, num_heads, head_dim) --> (3, bs, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv.unbind(0)

        attn_scores = Q @ K.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:seq_length, :seq_length], -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / K.shape[-1] ** -0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        output = attn_weights @ V
        output = output.transpose(1, 2)
        output = output.contiguous().view(bs, seq_length, embedding_dimension)
        output = self.proj(output)
        return output

# if __name__ == '__main__':
#     batch_size = 8
#     context_len = 256
#     embed_dim = 768
#     num_heads = 12
#     embeddings = torch.randn((batch_size, context_len, embed_dim), device = 'cuda' if torch.cuda.is_available else 'cpu')

#     attn_head = MultiHeadAttention(embed_dim, embed_dim, num_heads, context_len).to(device = 'cuda' if torch.cuda.is_available else 'cpu')
#     print(attn_head(embeddings), attn_head(embeddings).shape)
