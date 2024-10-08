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
        "--epochs",
        type=int,
        default=4,
        help="number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=3,
        help="train batch size",
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
        "--eval-freq",
        type=int,
        default=0, #200
        help="eval frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100,
        help="checkpoint frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/temp", #"checkpoints/diversity_v1_4"
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--val_loss_dir",
        type=str,
        default="val_losses/kfold",
        help="directory to save validation losses",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default='/oscar/data/stellex/shared/rt1-checkpoints/checkpoints/bridge/checkpoint_14400_loss_70.621.pt', #NOTE: include the path to load the checkpoint here
        help="checkpoint to load from; defaults to None",
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
        "--split-type",
        default = 'k_fold_scene',
        choices =  ['k_fold_scene', 'task_split', 'diversity_ablation', 'low_high_scene', 'cluster'],
    )
    parser.add_argument(
        '--low_div', 
        help='low diversity if true, else high', 
        action='store_true'
    )
    parser.add_argument(
        "--num-diversity-scenes",
        default = 4,
    )
    parser.add_argument(
        "--max-diversity-trajectories",
        default = 100,
    )
    parser.add_argument(
        "--train-subbatch",
        default=8,
    )
    parser.add_argument(
        "--eval-subbatch",
        default=5,
    )
    return parser.parse_args()


def main():

    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.wandb:
        wandb.init(project="rt1-data-diversity-v1", config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading dataset...")

    dataset_manager = DatasetManager(args.eval_scene, 0.8, 0.1, 0.1, split_style = args.split_type, diversity_scenes = args.num_diversity_scenes, max_trajectories = args.max_diversity_trajectories, low_div=args.low_div)
    
    if args.wandb and args.split_type == 'diversity_ablation':
        wandb.log({"task_keys": dataset_manager.train_dataset.dataset_keys})
    
    train_dataloader = DataLoader(dataset_manager.train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, collate_fn= dataset_manager.collate_batches, drop_last = False)
    val_dataloader = DataLoader(dataset_manager.val_dataset, batch_size = args.eval_batch_size, shuffle=True, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)
    

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

    print("Building policy...")
    policy = RT1Policy(
        observation_space=observation_space,
        action_space=action_space,
        device=args.device,
        checkpoint_path=args.load_checkpoint,
    )
    
    # Freeze all layers except the last one
    for name, param in policy.model.named_parameters():
        if "to_logits" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Verify that only the last layer is trainable
    # for name, param in policy.model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    
    policy.model.train()
    optimizer = Adam(policy.model.parameters(), lr=args.lr)

    #NOTE: has to be Not None because of raw instruction input
    
    text_embedding_model = (
        SentenceTransformer(args.sentence_transformer)
        if args.sentence_transformer
        else hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
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

    def get_text_embedding(observation: Dict):
        
        if args.sentence_transformer is not None:
            return text_embedding_model.encode(observation)
        else:
            embedded_observation = []

            for i in range(0, observation.shape[1]):
                
                try:
                    embedded_observation.append( np.array(text_embedding_model(observation[:, i]) ) )
                except:
                    print('EMBEDDING FAILED!')
                    # breakpoint()
                    
            # try:
            embedded_observation = np.stack(embedded_observation, axis=1)
            # except:
                # breakpoint()
            return embedded_observation

    print("Training...")
    num_batches = 0
    total_train_steps = 0
    total_val_steps = 0

    
    
    best_val_loss  = np.inf
    for epoch in range(args.epochs):
        print("STARTING EPOCH {}".format(epoch+1))
        
        for batch, train_batch in enumerate(train_dataloader):
            
            batch_steps = train_batch[0].shape[0]

            for idx in range(0, batch_steps, int(args.train_subbatch)):
                
                policy.model.train()
                
                num_batches += 1
                
                try:
                    res = get_text_embedding(train_batch[1][idx : min(idx + int(args.train_subbatch), batch_steps)])
                except:
                    breakpoint()
                observations = {
                    "image": train_batch[0][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    "context": res,
                }

                actions = {
                    'terminate_episode': train_batch[2][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'pickup_release': train_batch[3][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'body_position_delta': train_batch[4][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'body_yaw_delta': train_batch[5][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'body_pitch_delta': train_batch[6][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'arm_position_delta': train_batch[7][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'control_mode': train_batch[8][idx : min(idx + int(args.train_subbatch), batch_steps)]
                }

                padding = train_batch[9][idx : min(idx + int(args.train_subbatch), batch_steps)]
                total_train_steps += batch_steps

                
                loss, loss_std = policy.loss(observations, actions)

                if args.wandb:
                    print(f"Train loss Batch {num_batches}: {loss.item()} ± {loss_std.item()}")
                    wandb.log({"train_loss": loss.item(), "train_loss_std": loss_std.item()}, step= total_train_steps)
                else:
                    print(f"Train loss Batch {num_batches}: {loss.item()}")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                observations = {}; actions = {}

                
                if args.eval_freq and num_batches % args.eval_freq == 0:
                    
                    # Clear cache and collected garbage
                    gc.collect()
                    torch.cuda.empty_cache()

                    total_eval_loss = 0
                    total_eval_loss_std = 0
                    total_eval_count = 0
                    


                    print("Evaluating...")
                    for batch, val_batch in enumerate(val_dataloader):
                        
                        batch_steps_val = val_batch[0].shape[0]

                        print(f'Section {batch+1} of {len(val_dataloader)}')

                        for idx in tqdm(range(0, batch_steps_val, args.eval_subbatch)):

                            
                            policy.model.eval()
                            

                            total_val_steps += val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps_val)].shape[0]
                            total_eval_count += val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps_val)].shape[0]
                            
                            try:
                                res = get_text_embedding(val_batch[1][idx : min(idx + args.eval_subbatch, batch_steps_val)])
                            except:
                                breakpoint()
                            observations = {
                                "image": val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                "context": res,
                            }


                            actions = {
                                'terminate_episode': val_batch[2][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                'pickup_release': val_batch[3][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                'body_position_delta': val_batch[4][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                'body_yaw_delta': val_batch[5][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                'body_pitch_delta': val_batch[6][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                'arm_position_delta': val_batch[7][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                'control_mode': val_batch[8][idx : min(idx + args.eval_subbatch, batch_steps_val)]
                            }
                            
                            padding = val_batch[9][idx : min(idx + args.eval_subbatch, batch_steps_val)]

                            eval_loss, eval_loss_std = policy.loss(observations, actions)
                            
                            
                            total_eval_loss += eval_loss.item()*observations['image'].shape[0]
                            total_eval_loss_std += np.power(eval_loss_std.item(), 2)*observations['image'].shape[0]
                    
                    if args.wandb:
                        wandb.log(
                            {"eval_loss": total_eval_loss/total_eval_count, "eval_loss_std": np.sqrt(total_eval_loss_std/total_eval_count)},
                            step=total_train_steps,
                        )
                        print(f"Eval loss Batch {num_batches}: {total_eval_loss/total_eval_count}")
                    else:
                        val_dic = {}
                        eval_loss = total_eval_loss/total_eval_count
                        print(f"Eval loss Batch {num_batches}: {eval_loss}")
                        if eval_loss < best_val_loss:
                            best_val_loss = eval_loss
                            val_dic['best_val_loss'] = eval_loss
                        else:
                            val_dic['best_val_loss'] = best_val_loss
                        val_dic['curr_val_loss'] = eval_loss


                    os.makedirs(args.val_loss_dir, exist_ok=True)
                    with open(f'{args.val_loss_dir}/val_loss_batch_{num_batches}.json', 'w') as json_file:
                        json.dump(val_dic, json_file, indent=4)

                    
                if args.checkpoint_freq and num_batches % args.checkpoint_freq == 0:
                    checkpoint_path = (
                        f"{args.checkpoint_dir}/checkpoint_"
                        + f"{total_train_steps}"
                        + f"_loss_{loss.item():.3f}.pt"
                    )
                    torch.save(policy.model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
        
        print("FINISHED EPOCH {}".format(epoch+1))
    print("finished training")

if __name__ == "__main__":
    main()
