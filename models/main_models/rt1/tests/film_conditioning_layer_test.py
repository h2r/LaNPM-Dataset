"""Tests for film_conditioning_layer."""
import torch
from absl.testing import absltest, parameterized

from rt1_pytorch.film_efficientnet.film_conditioning_layer import FilmConditioning


class FilmConditioningLayerTest(parameterized.TestCase):
    @parameterized.parameters([2, 4])
    def test_film_conditioning_rank_two_and_four(self, conv_rank):
        batch = 2
        num_channels = 3
        embedding_dim = 512
        if conv_rank == 2:
            conv_layer = torch.randn(size=(batch, num_channels))
        elif conv_rank == 4:
            conv_layer = torch.randn(size=(batch, 1, 1, num_channels))
        else:
            raise ValueError(f"Unexpected conv rank: {conv_rank}")
        context = torch.rand(batch, embedding_dim)
        film_layer = FilmConditioning(embedding_dim, num_channels)
        out = film_layer(conv_layer, context)
        assert len(out.shape) == conv_rank


if __name__ == "__main__":
    absltest.main()