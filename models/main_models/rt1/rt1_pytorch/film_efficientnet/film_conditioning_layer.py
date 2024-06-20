import torch
from torch import nn


class FilmConditioning(nn.Module):
    def __init__(self, embedding_dim, num_channels):
        super().__init__()
        self._projection_add = nn.Linear(embedding_dim, num_channels)
        self._projection_mult = nn.Linear(embedding_dim, num_channels)
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        # From the paper
        nn.init.zeros_(self._projection_add.weight)
        nn.init.zeros_(self._projection_mult.weight)
        nn.init.zeros_(self._projection_add.bias)
        nn.init.zeros_(self._projection_mult.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        assert len(context.shape) == 2, f"Unexpected context shape: {context.shape}"
        assert (
            context.shape[1] == self.embedding_dim
        ), f"Unexpected context shape: {context.shape}"
        assert (
            x.shape[0] == context.shape[0]
        ), f"x and context must have the same batch size, but got {x.shape} and {context.shape}"
        projected_cond_add = self._projection_add(context)
        projected_cond_mult = self._projection_mult(context)

        if len(x.shape) == 4:
            projected_cond_add = projected_cond_add.unsqueeze(2).unsqueeze(3)
            projected_cond_mult = projected_cond_mult.unsqueeze(2).unsqueeze(3)
        else:
            assert len(x.shape) == 2

        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * x + projected_cond_add
        return result