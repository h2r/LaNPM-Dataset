import argparse
import os
from typing import Dict
import pdb
import gymnasium as gym
import numpy as np
import torch
import wandb
from sentence_transformers import SentenceTransformer
from torch.optim import Adam
import tensorflow_hub as hub 
from data import create_dataset
from rt1_pytorch.rt1_policy import RT1Policy
from tqdm import tqdm
from lanmp_dataloader.rt1_dataloader import DatasetManager, DataLoader
import gc
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-split",
        type=str,
        default="train[:-1000]",
        help="use e.g. train[:100] for the first 100 episodes",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="train[-1000:]",
        help="use e.g. eval[:100] for the first 100 episodes",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=3,
        help="eval batch size",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=6,
        help="number of frames per trajectory",
    )
    parser.add_argument(
        "--sentence-transformer",
        type=str,
        default=None,
        help="SentenceTransformer to use; default is None for original USE embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for training",
    )
    parser.add_argument(
        "--val_loss_dir",
        type=str,
        default="val_losses/kfold",
        help="directory to save validation losses",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/scene4",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="use wandb for logging",
        default=False,
    )

    parser.add_argument(
        "--eval-scene",
        default=4,
        help = "scene used as validation during k-fold cross validation",
    )
    parser.add_argument(
        "--eval-subbatch",
        default=5,
    )
    parser.add_argument(
        "--split-type",
        default = 'k_fold_scene',
        choices = ['k_fold_scene', 'task_split', 'diversity_ablation'],
    )

    parser.add_argument(
        "--num-diversity-scenes",
        default = 4,
    )

    parser.add_argument(
        "--max-diversity-trajectories",
        default = 100,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    

    os.makedirs(args.checkpoint_path, exist_ok=True)

    if args.wandb:
        wandb.init(project="rt1-finetuning", config=vars(args))

    os.makedirs(args.checkpoint_path, exist_ok=True)

    assert(len(os.listdir(args.checkpoint_path)) > 0 , "ERROR: checkpoint path is empty and has no saved checkpoints")

    print("Loading dataset...")
    
    
    dataset_manager = DatasetManager(args.eval_scene, 0.8, 0.1, 0.1, split_style = args.split_type, diversity_scenes = args.num_diversity_scenes, max_trajectories = args.max_diversity_trajectories)
    val_dataloader = DataLoader(dataset_manager.val_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)
    

    observation_space = gym.spaces.Dict(
        image=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3)),
        context=gym.spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
    )

    action_space = gym.spaces.Dict(

        body_yaw_delta = gym.spaces.Box(
            low= 0, #train_dataloader.body_orientation_lim['min_yaw']
            high= 255, #train_dataloader.body_orientation_lim['max_yaw']
            shape=(1,), 
            dtype=int
        ),

        body_pitch_delta = gym.spaces.Discrete(3),

        terminate_episode=gym.spaces.Discrete(2),

        pickup_release = gym.spaces.Discrete(3),

        body_position_delta = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (3,),
            dtype = np.int32
        ),

        arm_position_delta = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (3,),
            dtype = np.int32
        ),

        control_mode = gym.spaces.Discrete(7),
       
    )



    #NOTE: has to be Not None because of raw instruction input
    
    text_embedding_model = (
        SentenceTransformer(args.sentence_transformer)
        if args.sentence_transformer
        else hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
    )
    
    
   

    def get_text_embedding(observation: Dict):
        
        if args.sentence_transformer is not None:
            return text_embedding_model.encode(observation)
        else:
            embedded_observation = []

            for i in range(0, observation.shape[1]):
                
                try:
                    embedded_observation.append( np.array(text_embedding_model(observation[:, i]) ) )
                except:
                    pdb.set_trace()

            embedded_observation = np.stack(embedded_observation, axis=1)
            return embedded_observation

    
    def extract_train_step(filepath):
        return int(filepath.split('_')[1])


    print("Evaluating...")

    best_val_loss  = np.inf
    for idx, checkpoint_file in enumerate(list(sorted(os.listdir(args.checkpoint_path), key=extract_train_step))):
        print(f'Evaluating file: {idx} of {len(os.listdir(args.checkpoint_path))}')
        total_train_steps = int(checkpoint_file.split('_')[1])
        total_val_steps = 0

        
        total_eval_loss = 0
        total_eval_loss_std = 0
        total_eval_count = 0

        print("Building policy...")
        policy = RT1Policy(
            observation_space=observation_space,
            action_space=action_space,
            device=args.device,
            checkpoint_path=os.path.join(args.checkpoint_path, checkpoint_file),
        )
        
        

        # Total number of params
        total_params = sum(p.numel() for p in policy.model.parameters())
        # Transformer params
        transformer_params = sum(p.numel() for p in policy.model.transformer.parameters())
        # FiLM-EfficientNet and TokenLearner params
        tokenizer_params = sum(p.numel() for p in policy.model.image_tokenizer.parameters())
        print(f"Total params: {total_params}")
        print(f"Transformer params: {transformer_params}")
        print(f"FiLM-EfficientNet+TokenLearner params: {tokenizer_params}")


        for batch, val_batch in enumerate(val_dataloader):

            batch_steps = val_batch[0].shape[0]

            print(f'Section {batch+1} of {len(val_dataloader)}')

            for idx in tqdm(range(0, batch_steps, args.eval_subbatch)):

                
                policy.model.eval()
                
                
                total_val_steps += val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps)].shape[0]
                total_eval_count += val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps)].shape[0]

                observations = {
                    "image": val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps)],
                    "context": get_text_embedding(val_batch[1][idx : min(idx + args.eval_subbatch, batch_steps)]),
                }


                actions = {
                    'terminate_episode': val_batch[2][idx : min(idx + args.eval_subbatch, batch_steps)],
                    'pickup_release': val_batch[3][idx : min(idx + args.eval_subbatch, batch_steps)],
                    'body_position_delta': val_batch[4][idx : min(idx + args.eval_subbatch, batch_steps)],
                    'body_yaw_delta': val_batch[5][idx : min(idx + args.eval_subbatch, batch_steps)],
                    'body_pitch_delta': val_batch[6][idx : min(idx + args.eval_subbatch, batch_steps)],
                    'arm_position_delta': val_batch[7][idx : min(idx + args.eval_subbatch, batch_steps)],
                    'control_mode': val_batch[8][idx : min(idx + args.eval_subbatch, batch_steps)]
                }
                
                padding = val_batch[9][idx : min(idx + args.eval_subbatch, batch_steps)]

                eval_loss, eval_loss_std = policy.loss(observations, actions)
                
                
                total_eval_loss += eval_loss.item()*observations['image'].shape[0]
                total_eval_loss_std += np.power(eval_loss_std.item(), 2)*observations['image'].shape[0]
        
        if args.wandb:
            wandb.log(
                {"eval_loss": total_eval_loss/total_eval_count, "eval_loss_std": np.sqrt(total_eval_loss_std/total_eval_count)},
                step=total_train_steps,
            )
            print(f"Eval loss Step {total_train_steps}: {total_eval_loss/total_eval_count}")
        else:
            print(f"Eval loss Step {total_train_steps}: {total_eval_loss/total_eval_count}")
            val_dic = {}
            eval_loss = total_eval_loss/total_eval_count
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                val_dic['best_val_loss'] = eval_loss
            else:
                val_dic['best_val_loss'] = best_val_loss
            val_dic['curr_val_loss'] = eval_loss
            val_dic['checkpoint_name'] = checkpoint_file

        os.makedirs(f'{args.val_loss_dir}/{checkpoint_file}', exist_ok=True)
        with open(f'{args.val_loss_dir}/{checkpoint_file}/val_loss.json', 'w') as json_file:
            json.dump(val_dic, json_file, indent=4)

if __name__ == "__main__":
    main()
