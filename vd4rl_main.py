import argparse
import os
from typing import Dict, Iterable

import gymnasium as gym
import h5py
import numpy as np
import requests
import torch
import tqdm
import wandb
from dmc2gymnasium import DMCGym
from sentence_transformers import SentenceTransformer
from torch.optim import Adam

from rt1_pytorch.rt1_policy import RT1Policy

DATASET_URL = "https://huggingface.co/datasets/conglu/vd4rl/resolve/main/vd4rl/main/{domain}_{task}/expert/84px/{index}_{domain}_{task}_expert.hdf5"
ACTION_REPEAT = 2


class VD4RLEnv(gym.Env):
    def __init__(
        self,
        env_id: str,
        embedding: np.ndarray,
        embedding_dim: int,
        num_frames: int,
        dataset_dir: str,
    ):
        super().__init__()
        self.domain, self.task = env_id.split("-")
        self.env = DMCGym(self.domain, self.task)
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.num_frames = num_frames
        self._load_dataset(dataset_dir)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
                ),
                "embedding": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(self.embedding_dim,), dtype=np.float32
                ),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Dict({"action_key": self.env.action_space})

    def reset(self):
        _, info = self.env.reset()
        obs = self.env.render(84, 84)
        return ({"image": obs, "embedding": self.embedding}, info)

    def step(self, action):
        action = action["action_key"]
        term = False
        trunc = False
        for _ in range(ACTION_REPEAT):
            _, r, term, trunc, info = self.env.step(action)
            if term or trunc:
                break
        o = self.env.render(84, 84)
        return ({"image": o, "embedding": self.embedding}, r, term, trunc, info)

    def _load_dataset(self, dataset_dir: str):
        os.makedirs(dataset_dir, exist_ok=True)
        observations = []
        actions = []
        for index in tqdm.trange(4):
            file = f"{index}_{self.domain}_{self.task}_expert.hdf5"
            path = os.path.join(dataset_dir, file)
            if not os.path.exists(path):
                url = DATASET_URL.format(
                    domain=self.domain,
                    task=self.task,
                    index=index,
                )
                if self.domain == "humanoid" and self.task == "walk":
                    url = url.rsplit("/")[0] + f"/{index}_expert.hdf5"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(response.content)
            with h5py.File(path, "r") as f:
                observations.append(f["observation"][:])
                actions.append(f["action"][:])
        self.observations = np.concatenate(observations)
        self.actions = np.concatenate(actions)

    def get_dataset(self, batch_size: int) -> Iterable[Dict]:
        # We expect self.num_frames trajectories per episode
        num_episodes = np.ceil(batch_size / self.num_frames).astype(int)
        # Leftover trajectories from last episode
        prev_obs = None
        prev_act = None
        for idx in range(0, self.actions.shape[0], num_episodes * 501):
            # Get `batch_size` number of episodes
            obs = self.observations[idx : idx + num_episodes * 501]
            act = self.actions[idx : idx + num_episodes * 501]

            # Convert to (b, t, ...)
            obs = np.reshape(obs, (num_episodes, 501, *obs.shape[1:]))
            act = np.reshape(act, (num_episodes, 501, *act.shape[1:]))

            # drop the last timestep and action from each episode
            obs = obs[:, :-1]
            act = act[:, :-1]

            # frame-stack by rolling self.num_frames times over t
            num_traj = 500 - self.num_frames + 1
            indices = np.stack(
                [np.arange(s, s + num_traj) for s in range(self.num_frames)],
                axis=-1,
            )

            # (b, t, ...) -> (b, t - f + 1, f, ...)
            obs = np.take(obs, indices, axis=1)
            act = np.take(act, indices, axis=1)

            # (b, t - f + 1, f, ...) -> (b * (t - f + 1), f, ...)
            obs = np.reshape(obs, (num_episodes * num_traj, *obs.shape[2:]))
            act = np.reshape(act, (num_episodes * num_traj, *act.shape[2:]))

            # Concatenate with leftover trajectories from last episode
            if prev_obs is not None:
                obs = np.concatenate([prev_obs, obs], axis=0)
                act = np.concatenate([prev_act, act], axis=0)

            for batch in range(0, obs.shape[0], batch_size):
                if batch + batch_size > obs.shape[0]:
                    # Save leftover trajectories and break
                    prev_obs = obs[batch:]
                    prev_act = act[batch:]
                    break

                yield {
                    "observation": {
                        "image": obs[batch : batch + batch_size],
                        "embedding": np.tile(
                            np.expand_dims(self.embedding, (0, 1)),
                            (batch_size, self.num_frames, 1),
                        ),
                    },
                    "action": {"action_key": act[batch : batch + batch_size]},
                }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="walker-walk",
        help="name of the environment",
        choices=[
            "walker-walk",
            "cheetah-run",
            "humanoid-walk",
        ],
    )
    parser.add_argument(
        "--context",
        type=str,
        default="""Move forward by walking upright on two legs, 
        while maintaining balance and stability""",
    )
    # cheetah-run: """Run forward rapidly on all four legs,
    # coordinating movements for speed and efficiency"""
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size in number of trajectories",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=4,
        help="number of frames per trajectory",
    )
    parser.add_argument(
        "--sentence-transformer",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer to use for text embedding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for training",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="eval frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="checkpoint frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/vd4rl",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="checkpoint to load from; defaults to None",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets",
        help="local directory for datasets",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="use wandb for logging",
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.wandb:
        wandb.init(project="rt1-vd4rl", config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    text_embedding_model = SentenceTransformer(args.sentence_transformer)
    embedding_dim = text_embedding_model.get_sentence_embedding_dimension()
    embedding = text_embedding_model.encode(args.context)

    print("Loading dataset...")
    env = VD4RLEnv(
        env_id=args.env,
        embedding=embedding,
        embedding_dim=embedding_dim,
        num_frames=args.trajectory_length,
        dataset_dir=args.dataset_dir,
    )

    print("Building policy...")
    policy = RT1Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        arch="efficientnet_b0",
        action_bins=512,
        num_layers=4,
        num_heads=4,
        feed_forward_size=512,
        dropout_rate=0.01,
        time_sequence_length=args.trajectory_length,
        embedding_dim=embedding_dim,
        use_token_learner=True,
        token_learner_bottleneck_dim=32,
        token_learner_num_output_tokens=8,
        device=args.device,
        checkpoint_path=args.load_checkpoint,
    )
    policy.model.train()
    optimizer = Adam(policy.model.parameters(), lr=args.lr)
    # Total number of params
    total_params = sum(p.numel() for p in policy.model.parameters())
    # Transformer params
    transformer_params = sum(p.numel() for p in policy.model.transformer.parameters())
    # FiLM-EfficientNet and TokenLearner params
    tokenizer_params = sum(p.numel() for p in policy.model.image_tokenizer.parameters())
    print(f"Total params: {total_params}")
    print(f"Transformer params: {transformer_params}")
    print(f"FiLM-EfficientNet+TokenLearner params: {tokenizer_params}")

    def get_text_embedding(observation: Dict):
        return observation["embedding"]

    print("Training...")
    num_batches = 0
    for epoch in range(1, args.epochs + 1):
        train_dataset = env.get_dataset(batch_size=args.batch_size)
        for batch in train_dataset:
            policy.model.train()
            num_batches += 1
            observations = {
                "image": batch["observation"]["image"],
                "context": get_text_embedding(batch["observation"]),
            }
            actions = batch["action"]
            loss = policy.loss(observations, actions)
            if args.wandb:
                wandb.log(
                    {"train_loss": loss.item()},
                    step=num_batches * args.batch_size,
                )
            else:
                print(f"Batch {num_batches} train loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.eval_freq and num_batches % args.eval_freq == 0:
                print("Evaluating...")
                policy.model.eval()
                obs, _ = env.reset()
                obs_stacked = {
                    k: np.stack([v for _ in range(args.trajectory_length)])
                    for k, v in obs.items()
                }
                observations = {"image": [], "context": []}
                actions = {"action_key": []}
                term = False
                trunc = False
                reward = 0.0
                ts = 0
                while not (term or trunc):
                    cur_obs = {
                        "image": obs_stacked["image"],
                        "context": get_text_embedding(obs_stacked),
                    }

                    # add batch dimension
                    cur_obs["image"] = np.expand_dims(cur_obs["image"], axis=0)
                    cur_obs["context"] = np.expand_dims(cur_obs["context"], axis=0)

                    act = policy.act(cur_obs)

                    # remove batch dimension
                    act = {k: v[0] for k, v in act.items()}
                    new_obs, rew, term, trunc, info = env.step(act)
                    obs_stacked = {
                        k: np.concatenate(
                            [
                                obs_stacked[k][1:],
                                np.expand_dims(new_obs[k], axis=0),
                            ]
                        )
                        for k in new_obs.keys()
                    }
                    observations["image"].append(obs_stacked["image"])
                    observations["context"].append(get_text_embedding(obs_stacked))
                    actions["action_key"].append(act["action_key"])
                    reward += rew * (info["discount"] ** ts)
                    ts += 1
                if args.wandb:
                    wandb.log(
                        {"eval_return": reward},
                        step=num_batches * args.batch_size,
                    )
                else:
                    print(f"Batch {num_batches} eval return: {reward}")
            if args.checkpoint_freq and num_batches % args.checkpoint_freq == 0:
                checkpoint_path = (
                    f"{args.checkpoint_dir}/checkpoint_"
                    + f"{num_batches * args.batch_size * epoch}"
                    + f"_loss_{loss.item():.3f}.pt"
                )
                torch.save(policy.model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()