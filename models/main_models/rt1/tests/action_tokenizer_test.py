"""Tests for action_tokenizer."""
import unittest

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from rt1_pytorch.tokenizers.action_tokenizer import RT1ActionTokenizer


class ActionTokenizerTest(unittest.TestCase):
    def testTokenize_int32(self):
        action_space = Dict(terminate_episode=Discrete(2))
        tokenizer = RT1ActionTokenizer(action_space, action_bins=10)
        self.assertEqual(1, tokenizer.tokens_per_action)
        action = dict(terminate_episode=np.array([1], dtype=np.int32))
        action_tokens = tokenizer.tokenize(action)
        self.assertEqual(action["terminate_episode"], action_tokens)

    def testTokenize_int32_out_of_bounds(self):
        action_space = Dict(terminate_episode=Discrete(2))
        tokenizer = RT1ActionTokenizer(action_space, action_bins=10)
        self.assertEqual(1, tokenizer.tokens_per_action)
        action = dict(terminate_episode=np.array([3], dtype=np.int32))
        with self.assertRaises(ValueError):
            tokenizer.tokenize(action)

    def testDetokenize_int32(self):
        action_space = Dict(terminate_episode=Discrete(2))
        tokenizer = RT1ActionTokenizer(action_space, action_bins=10)
        action = tokenizer.detokenize(np.array([0], dtype=np.int32))
        self.assertEqual(action["terminate_episode"], np.array([0]))
        # OOV 3 token should become a default one hot: [1, 0]
        action = tokenizer.detokenize(np.array([3], dtype=np.int32))
        self.assertEqual(action["terminate_episode"], np.array([0]))

    def testTokenize_float(self):
        action_space = Dict(
            world_vector=Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        )
        tokenizer = RT1ActionTokenizer(action_space, action_bins=10)
        self.assertEqual(3, tokenizer.tokens_per_action)
        action = dict(world_vector=[0.1, 0.5, -0.8])
        action_tokens = tokenizer.tokenize(action)
        self.assertSequenceEqual([4, 6, 0], list(action_tokens.tolist()))

    def testTokenize_float_with_time_dimension(self):
        action_space = Dict(
            world_vector=Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        )
        tokenizer = RT1ActionTokenizer(action_space, action_bins=10)
        self.assertEqual(3, tokenizer.tokens_per_action)
        batch_size = 2
        time_dimension = 3
        action = dict(
            world_vector=np.array(
                [
                    [0.1, 0.5, -0.8],
                    [0.1, 0.5, -0.8],
                    [0.1, 0.5, -0.8],
                    [0.1, 0.5, -0.8],
                    [0.1, 0.5, -0.8],
                    [0.1, 0.5, -0.8],
                ],
            ).reshape((batch_size, time_dimension, 3)),
        )
        action_tokens = tokenizer.tokenize(action)
        self.assertSequenceEqual(
            [batch_size, time_dimension, tokenizer.tokens_per_action],
            action_tokens.shape,
        )

    def testTokenize_float_at_limits(self):
        minimum = -1.0
        maximum = 1.0
        action_bins = 10
        action_space = Dict(
            world_vector=Box(low=minimum, high=maximum, shape=(2,), dtype=np.float32)
        )
        tokenizer = RT1ActionTokenizer(action_space, action_bins=action_bins)
        self.assertEqual(2, tokenizer.tokens_per_action)
        action = dict(world_vector=[minimum, maximum])
        action_tokens = tokenizer.tokenize(action)
        # Minimum value will go to 0
        # Maximum value witll go to action_bins-1
        self.assertSequenceEqual([0, action_bins - 1], action_tokens.tolist())

    def testTokenize_invalid_action_space_shape(self):
        action_space = Dict(
            world_vector=Box(low=-1.0, high=1.0, shape=(2, 2), dtype=np.float32)
        )
        with self.assertRaises(ValueError):
            RT1ActionTokenizer(action_space, action_bins=10)

    def testTokenizeAndDetokenizeIsEqual(self):
        action_space = Dict(
            world_vector=Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            rotation_delta=Box(
                low=-np.pi / 2.0, high=np.pi / 2.0, shape=(3,), dtype=np.float32
            ),
            gripper_closedness_action=Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            terminate_episode=Discrete(3),
        )

        tokenizer = RT1ActionTokenizer(
            action_space,
            action_bins=256,
            action_order=[
                "terminate_episode",
                "world_vector",
                "rotation_delta",
                "gripper_closedness_action",
            ],
        )
        self.assertEqual(8, tokenizer.tokens_per_action)

        # Repeat the following test N times with fuzzy inputs.
        n_repeat = 10
        for _ in range(n_repeat):
            action = dict(
                world_vector=np.random.uniform(low=-1.0, high=1.0, size=3),
                rotation_delta=np.random.uniform(
                    low=-np.pi / 2.0, high=np.pi / 2.0, size=3
                ),
                gripper_closedness_action=np.random.uniform(low=0.0, high=1.0, size=1),
                terminate_episode=np.array(0, dtype=np.int32),
            )
            action_tokens = tokenizer.tokenize(action)
            policy_action = tokenizer.detokenize(action_tokens)

            for k in action:
                self.assertTrue(
                    np.allclose(action[k], policy_action[k], atol=1e-1),
                    f"Failed at {k} with {action[k]} != {policy_action[k]}.",
                )

            # Repeat the test with batched actions
            batched_action = dict(
                world_vector=[
                    np.random.uniform(low=-1.0, high=1.0, size=3),
                    np.random.uniform(low=-1.0, high=1.0, size=3),
                ],
                rotation_delta=[
                    np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=3),
                    np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=3),
                ],
                gripper_closedness_action=[
                    np.random.uniform(low=0.0, high=1.0, size=1),
                    np.random.uniform(low=0.0, high=1.0, size=1),
                ],
                terminate_episode=[0, 1],
            )
            action_tokens = tokenizer.tokenize(batched_action)
            policy_action = tokenizer.detokenize(action_tokens)

            for k in batched_action:
                for a, policy_a in zip(batched_action[k], policy_action[k]):
                    self.assertTrue(
                        np.allclose(a, policy_a, atol=1e-1),
                        f"Failed at {k} with {a} != {policy_a}.",
                    )


if __name__ == "__main__":
    unittest.main()