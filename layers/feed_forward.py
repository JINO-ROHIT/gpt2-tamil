import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension: int = 768, scaling_factor: int = 4, bias: bool = False, **kwargs):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.scaling_factor = scaling_factor
        self.bias = bias

        self.layers = nn.Sequential(
            nn.Linear(self.embedding_dimension, self.scaling_factor * self.embedding_dimension),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(self.scaling_factor * self.embedding_dimension, self.embedding_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__ == '__main__':
    ff = FeedForward()
    x = torch.randn(10, 10, 768)
    print(ff(x))
