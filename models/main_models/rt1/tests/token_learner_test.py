"""Tests for token_learner."""
import unittest

import torch

from rt1_pytorch.tokenizers.token_learner import TokenLearner


class TokenLearnerTest(unittest.TestCase):
    def testTokenLearner_h_w_split(self):
        batch = 5
        embedding_dim = 512
        num_tokens = 8
        device = "cuda" if torch.cuda.is_available() else "cpu"
        token_learner_layer = TokenLearner(
            embedding_dim=embedding_dim, num_tokens=num_tokens, device=device
        )

        inputvec = torch.randn((batch, embedding_dim, 10, 10), device=device)

        learnedtokens = token_learner_layer(inputvec)
        self.assertEqual(learnedtokens.shape, (batch, embedding_dim, num_tokens))

    def testTokenLearner_hw(self):
        batch = 5
        embedding_dim = 512
        num_tokens = 8
        device = "cuda" if torch.cuda.is_available() else "cpu"
        token_learner_layer = TokenLearner(
            embedding_dim=embedding_dim, num_tokens=num_tokens, device=device
        )

        inputvec = torch.randn((batch, embedding_dim, 100), device=device)

        learnedtokens = token_learner_layer(inputvec)
        self.assertEqual(learnedtokens.shape, (batch, embedding_dim, num_tokens))


if __name__ == "__main__":
    unittest.main()