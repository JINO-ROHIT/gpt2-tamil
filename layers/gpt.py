import torch
import torch.nn as nn
from loguru import logger

from layers import TransformerBlock


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TamilGPT model.
        
        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, context_length)
        
        Returns:
            torch.Tensor: Logits for next token prediction with shape (batch_size, context_length, vocab_size)
        """
        batch_size, context_length = x.shape

        # Ensure input doesn't exceed the model's context length
        assert context_length <= self.context_length, \
            f"Input sequence length {context_length} exceeds model's context length {self.context_length}"


        position_indices = torch.arange(context_length, device=x.device)
        token_emb = self.token_embedding(x)

        position_emb = self.positional_embedding(position_indices)

        x = token_emb + position_emb

        x = self.embedding_dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.norm(x)

        logits = self.language_model_head(x)

        return logits


# if __name__ == '__main__':
#     gpt = TamilGPT(vocab_size = 32000, embedding_dimension = 768, context_length = 256, num_heads = 12, scaling_factor = 4, num_layers = 12, bias = False, dropout = 0, weight_tying = True)
