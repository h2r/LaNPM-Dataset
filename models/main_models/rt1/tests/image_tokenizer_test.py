
"""Tests for image_tokenizer."""
import unittest

import torch
from absl.testing import parameterized

from rt1_pytorch.tokenizers.image_tokenizer import RT1ImageTokenizer

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


class ImageTokenizerTest(parameterized.TestCase):
    @parameterized.named_parameters(
        *[(f"sample_image_{m}", m, 512, 224, False, 8) for m in MODELS],
        *[(f"sample_image_token_learner_{m}", m, 512, 224, True, 8) for m in MODELS],
    )
    def testTokenize(
        self, arch, embedding_dim, image_resolution, use_token_learner, num_tokens
    ):
        batch = 4
        device = "cuda"
        tokenizer = RT1ImageTokenizer(
            arch=arch,
            embedding_dim=embedding_dim,
            use_token_learner=use_token_learner,
            token_learner_num_output_tokens=num_tokens,
            device=device,
        )

        image = torch.randn((batch, image_resolution, image_resolution, 3))
        image = torch.clip(image, 0.0, 1.0)
        image = image.to(device)
        context_vector = torch.FloatTensor(size=(batch, 512)).uniform_()
        context_vector = context_vector.to(device)
        image_tokens = tokenizer(image, context_vector)
        self.assertEqual(image_tokens.shape, (batch, 512, tokenizer.num_output_tokens))


if __name__ == "__main__":
    unittest.main()
