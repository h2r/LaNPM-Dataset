"""Pytorch implementation of TokenLearner(Ryoo et al 2021)."""

import torch
from torch import nn


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        out_dim: int,
        dropout_rate: float = 0.1,
        device="cuda",
    ):
        """Initializer for the MLP Block.

        This computes outer_dense(gelu(hidden_dense(input))), with dropout
        applied as necessary.

        Args:
          input_dim: The dimension of the input.
          mlp_dim: The dimension of the inner representation (output of hidden
            layer). Usually larger than the input/output dim.
          out_dim: The output dimension of the block.
          dropout_rate: Dropout rate to be applied after dense ( & activation)
            layers.
          device: The device to place the model on.
        """
        super().__init__()
        self._hidden_dropout = nn.Dropout(dropout_rate)
        self._output_dropout = nn.Dropout(dropout_rate)
        self._hidden_layer = nn.Linear(input_dim, mlp_dim, device=device)
        self._output_layer = nn.Linear(mlp_dim, out_dim, device=device)
        nn.init.xavier_uniform_(self._hidden_layer.weight)
        nn.init.xavier_uniform_(self._output_layer.weight)
        nn.init.normal_(self._hidden_layer.bias, std=1e-6)
        nn.init.normal_(self._output_layer.bias, std=1e-6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies Transformer MlpBlock module."""
        x = self._hidden_layer(inputs)
        x = nn.functional.gelu(x)
        x = self._hidden_dropout(x)
        x = self._output_layer(x)
        x = self._output_dropout(x)
        return x


class TokenLearner(nn.Module):
    """TokenLearner module V1.1 (https://arxiv.org/abs/2106.11297)."""

    def __init__(
        self,
        embedding_dim: int,
        num_tokens: int,
        bottleneck_dim: int = 64,
        dropout_rate: float = 0.0,
        device="cuda",
    ):
        super().__init__()

        self.layernorm = nn.LayerNorm(embedding_dim, eps=1e-6, device=device)
        self.mlp = MlpBlock(
            input_dim=embedding_dim,
            mlp_dim=bottleneck_dim,
            out_dim=num_tokens,
            dropout_rate=dropout_rate,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs.shape) == 4:
            bs, c, h, w = inputs.shape
            inputs = torch.reshape(inputs, [bs, c, h * w])
        inputs = inputs.permute(0, 2, 1)  # Shape: [bs, h*w, c]

        selected = self.layernorm(inputs)

        selected = self.mlp(selected)  # Shape: [bs, h*w, n_token].
        selected = nn.functional.softmax(selected, dim=-1)
        selected = selected.permute(0, 2, 1)  # Shape: [bs, n_token, h*w]

        feat = torch.einsum("...si,...id->...sd", selected, inputs)
        feat = feat.permute(0, 2, 1)

        return feat  # Shape: [bs, c, n_token]