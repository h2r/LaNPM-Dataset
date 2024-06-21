"""The image tokenizer combining the FiLMEfficientNet and TokenLearner from RT1.
"""
import torch
from torch import nn

from rt1_pytorch.film_efficientnet.film_conditioning_layer import FilmConditioning
from rt1_pytorch.film_efficientnet.film_efficientnet import FilmEfficientNet
from rt1_pytorch.tokenizers.token_learner import TokenLearner


class RT1ImageTokenizer(nn.Module):
    """Tokenizes based on vocab size."""

    def __init__(
        self,
        arch: str = "efficientnet_b3",
        embedding_dim: int = 512,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=8,
        dropout_rate=0.1,
        device="cuda",
    ):
        """Instantiates a RT1ImageTokenizer.

        Args:
          arch: The efficientnet variant to use.
          embedding_dim: The embedding size of the tokens.
          use_token_learner: Whether to use token learner. See
            https://arxiv.org/abs/2106.11297
          num_tokens: Relevant only for token learner - the number of learned
            tokens.
          token_learner_bottleneck_dim: Relevant only for token learner - the
            dimension of the bottleneck layer.
          token_learner_num_output_tokens: Relevant only for token learner -
            the number of output tokens.
          dropout_rate: Relevant only for token learner - the dropout rate.
          device: The device to place the model on.
        """
        super().__init__()

        self.film_efficientnet = FilmEfficientNet(
            arch=arch, embedding_dim=embedding_dim, device=device
        )
        self.num_output_tokens = self.film_efficientnet.output_hw**2

        self._use_token_learner = use_token_learner
        if self._use_token_learner:
            self._token_learner = TokenLearner(
                embedding_dim=embedding_dim,
                num_tokens=token_learner_num_output_tokens,
                bottleneck_dim=token_learner_bottleneck_dim,
                dropout_rate=dropout_rate,
                device=device,
            )
            self.num_output_tokens = token_learner_num_output_tokens

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Gets image tokens.

        Args:
          image: Images of shape (b, h, w, 3) to tokenize.
          context: A context vector (e.g., a natural language embedding).
            Expected to have shape (b, embedding_dim).

        Returns:
          tokens: has shape (batch, num_tokens_per_timestep, embedding_dim)
        """
        assert len(context.shape) == 2, f"Unexpected context shape: {context.shape}"

        tokens = self.film_efficientnet(image, context)
        if len(tokens.shape) == 4:
            # (b, c, h, w) -> (b, c, h*w)
            tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
        if self._use_token_learner:
            tokens = self._token_learner(tokens)
        return tokens