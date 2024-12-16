import torch
import torch.nn as nn
from loguru import logger

from layers import FeedForward, MultiHeadAttention, TransformerBlock

class TamilGPT(nn.Module):
    def __init__(self, vocab_size: int, embedding_dimension: int, context_length: int, num_heads: int, scaling_factor: int, num_layers: int, bias: bool, dropout: float, weight_tying: bool = True):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.context_length = context_length
        self.num_heads = num_heads
        self.scaling_factor = scaling_factor
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.weigh_tying = weight_tying

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dimension)
        self.positional_embedding = nn.Embedding(self.context_length, self.embedding_dimension)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dimension = self.embedding_dimension,
                    context_length = self.context_length,
                    num_heads = self.num_heads,
                    bias = self.bias,
                    dropout = self.dropout,
                    scaling_factor = self.scaling_factor,
                )
                for _ in range(self.num_layers)
            ],
        )
        self.norm = nn.LayerNorm(self.embedding_dimension, bias = self.bias)
        self.language_model_head = nn.Linear(self.embedding_dimension, self.vocab_size, bias=False)
        self.__tie_weight()

        logger.info(
            "GPT language model is created with number of parameters: {:.2f} million".format(
                self.__get_parameters_number() / 1e6,
            ),
        )
    
    def __tie_weight(self):
        self.token_embedding.weight = self.language_model_head.weight

    def __get_parameters_number(self) -> int:
        params_count = sum(param.numel() for param in self.parameters())
        return params_count


if __name__ == '__main__':
    gpt = TamilGPT(vocab_size = 32000, embedding_dimension = 768, context_length = 256, num_heads = 12, scaling_factor = 4, num_layers = 12, bias = False, dropout = 0, weight_tying = True)
