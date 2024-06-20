"""Tests for pretrained_efficientnet_encoder."""

import torch
from absl.testing import absltest, parameterized
from skimage import data

from rt1_pytorch.film_efficientnet.film_efficientnet import (
    FilmEfficientNet,
    decode_predictions,
)

MODELS = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    # "efficientnet_b4",
    # "efficientnet_b5",
    # "efficientnet_b6",
    # "efficientnet_b7",
    "efficientnet_v2_s",
    # "efficientnet_v2_m",
    # "efficientnet_v2_l",
]


class FilmEfficientNetTest(parameterized.TestCase):
    @parameterized.parameters(MODELS)
    def test_encoding(self, model_name):
        """Test that we get a correctly shaped encoding."""
        embedding_dim = 512
        batch_size = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = torch.tensor(data.chelsea()).repeat(batch_size, 1, 1, 1)
        context = torch.FloatTensor(size=(batch_size, embedding_dim)).uniform_(-1, 1)
        model = FilmEfficientNet(model_name, device=device).eval()
        image = image.to(device)
        context = context.to(device)
        preds = model(image, context)
        self.assertEqual(
            preds.shape, (batch_size, 512, model.output_hw, model.output_hw)
        )

    @parameterized.parameters(MODELS)
    def test_imagenet_classification(self, model_name):
        """Test that we can correctly classify an image of a cat."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = torch.tensor(data.chelsea())
        model = FilmEfficientNet(model_name, include_top=True, device=device).eval()
        image = image.to(device)
        preds = model(image)
        predicted_names = [n[0] for n in decode_predictions(preds, top=3)[0]]
        self.assertIn("tabby", predicted_names)


if __name__ == "__main__":
    absltest.main()