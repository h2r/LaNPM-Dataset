import torch
from absl.testing import absltest, parameterized

from rt1_pytorch.rt1_model import RT1Model


class RT1ModelTest(parameterized.TestCase):
    @parameterized.parameters(["cpu", "cuda"])
    def test_videos(self, device):
        model = RT1Model(device=device)

        batch_size = 1
        videos = torch.rand(batch_size, 6, 3, 224, 224, device=device)
        logits = model(videos)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (batch_size, 6, 11, 256))

    @parameterized.parameters(["cpu", "cuda"])
    def test_videos_and_texts(self, device="cpu"):
        model = RT1Model(device=device)

        batch_size = 1
        videos = torch.rand(batch_size, 6, 3, 224, 224, device=device)
        texts = torch.rand(batch_size, 6, 512, device=device)
        logits = model(videos, texts)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (batch_size, 6, 11, 256))

    @parameterized.parameters(["cpu", "cuda"])
    def test_videos_and_action_logits(self, device="cpu"):
        model = RT1Model(device=device)

        batch_size = 1
        videos = torch.rand(batch_size, 6, 3, 224, 224, device=device)
        action_logits = torch.rand(batch_size, 6, 11, 256, device=device)
        logits = model(videos, action_logits=action_logits)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (batch_size, 6, 11, 256))

    @parameterized.parameters(["cpu", "cuda"])
    def test_videos_and_texts_and_action_logits(self, device="cpu"):
        model = RT1Model(device=device)

        batch_size = 1
        videos = torch.rand(batch_size, 6, 3, 224, 224, device=device)
        texts = torch.rand(batch_size, 6, 512, device=device)
        action_logits = torch.rand(batch_size, 6, 11, 256, device=device)
        logits = model(videos, texts, action_logits)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (batch_size, 6, 11, 256))


if __name__ == "__main__":
    absltest.main()