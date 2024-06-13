import numpy as np
from absl.testing import absltest, parameterized
from gymnasium.spaces import Box, Dict, Discrete
from skimage import data

from rt1_pytorch.rt1_policy import RT1Policy


class RT1PolicyTest(parameterized.TestCase):
    @parameterized.parameters(["cpu", "cuda"])
    def test_policy_act_and_loss(self, device="cpu"):
        observation_space = Dict(
            image=Box(low=0, high=255, shape=(300, 451, 3), dtype=np.uint8),
            context=Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
        )
        action_space = Dict(
            world_vector=Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            base_displacement_vertical_rotation=Box(
                low=-np.pi / 2.0, high=np.pi / 2.0, shape=(1,), dtype=np.float32
            ),
            gripper_closedness_action=Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            terminate_episode=Discrete(3),
            base_displacement_vector=Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            ),
            rotation_delta=Box(
                low=-np.pi / 2.0, high=np.pi / 2.0, shape=(3,), dtype=np.float32
            ),
        )
        policy = RT1Policy(observation_space, action_space, device=device)

        image = data.chelsea()
        videos = np.reshape(image, (1, 1, *image.shape)).repeat(6, axis=1)
        # videos (b, f, h, w, c) = (1, 6, 300, 451, 3)
        context = np.random.rand(1, 6, 512).astype(np.float32)
        # context (b, f, d) = (1, 6, 512)
        observations = {"image": videos, "context": context}
        actions = policy.act(observations)

        action_tokens = policy.action_tokenizer.tokenize(actions)

        self.assertEqual(action_tokens.shape, (1, 12))
        obs = {k: v[0][0] for k, v in observations.items()}
        act = {k: v[0] for k, v in actions.items()}
        self.assertTrue(observation_space.contains(obs))
        self.assertTrue(action_space.contains(act))

        target_actions = {
            k: np.expand_dims(v, axis=1).repeat(6, axis=1) for k, v in actions.items()
        }

        loss = policy.loss(observations=observations, target_actions=target_actions)
        self.assertGreater(loss, 0)

    # TODO (Rohan138): Add more tests


if __name__ == "__main__":
    absltest.main()