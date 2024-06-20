import os
import sys

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
from PIL import Image
from tqdm import tqdm
# from models.utils.data_utils import split_data
import pdb
#mainly for debugging
import matplotlib.pyplot as plt
import numpy as np
import re
import json
import sys
from copy import copy
import random

sys.path.append('..')

DATASET_PATH = '/oscar/data/stellex/shared/lanmp/sim_dataset.hdf5'

'''
train_keys, val_keys, test_keys = split_data(self.args.data, splits['train'], splits['val'], splits['test'])
'''

def split_data(hdf5_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Assuming trajectories or data units are top-level groups in the HDF5 file
        keys = list(hdf_file.keys())
        total_items = len(keys)
        
        # Generate a shuffled array of indices
        indices = np.arange(total_items)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        train_end = int(train_ratio * total_items)
        val_end = train_end + int(val_ratio * total_items)
        
        # Split the indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Convert indices back to keys (assuming order in keys list is stable and matches original order)
        train_keys = [keys[i] for i in train_indices]
        val_keys = [keys[i] for i in val_indices]
        test_keys = [keys[i] for i in test_indices]
        
        return train_keys, val_keys, test_keys

def split_by_scene(hdf5_path):

    #mapping which keys are relevant to specific scenes
    scene_to_keys = {}

    with h5py.File(hdf5_path, 'r') as hdf_file:

        keys = list(hdf_file.keys())

        for k in keys:
            traj_json_dict = json.loads(hdf_file[k]['folder_0'].attrs['metadata'])

            if traj_json_dict['scene'] not in scene_to_keys:
                scene_to_keys[traj_json_dict['scene']] = []
            
            scene_to_keys[traj_json_dict['scene']].append(k)
    
    for k in scene_to_keys.keys():
        scene_to_keys[k] = list(sorted(scene_to_keys[k]))
    
    with open('./lanmp_dataloader/scene_to_keys.json', 'w') as f:
        json.dump(scene_to_keys, f)

    return scene_to_keys



def sort_folders(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

class DatasetManager(object):

    '''
    NOTE: kwargs should contain a dictionary with keys {'train_split' : x, 'val_split': y, 'test_split':z} where x+y+z = 1
    '''
    def __init__(self, val_scene=1, train_split=0.8, val_split=0.1, test_split=0.1, split_style='task_split', diversity_scenes=1, max_trajectories=100):

        assert( train_split + val_split + test_split == 1.0, 'Error: train, val and test split do not sum to 1.0')

        
        #train_keys, val_keys, test_keys = split_data(DATASET_PATH, train_split, val_split, test_split)
        if 'scene_to_keys.json' not in os.listdir('./lanmp_dataloader'):
            self.scene_to_keys = split_by_scene(DATASET_PATH)
        else:
            with open('./lanmp_dataloader/scene_to_keys.json') as f:
                self.scene_to_keys = json.load(f)
             

        self.scenes = list(sorted(list(self.scene_to_keys.keys())))

        assert( split_style in ['k_fold_scene', 'task_split', 'diversity_ablation'], "Error: input split_style is invalid")

        if split_style == 'k_fold_scene':
            assert( val_scene < len(self.scenes), "Error: input scene is out of index space")
            train_keys = []
            for x in range(0, len(self.scenes)):
                if x!=val_scene:
                    train_keys += self.scene_to_keys[self.scenes[x]]  

            val_keys = self.scene_to_keys[self.scenes[val_scene]]
            test_keys = None

        elif split_style == 'task_split':

            train_keys = []
            val_keys = []

            for scene in self.scenes:
                
                scene_keys = copy(self.scene_to_keys[scene])
                random.shuffle(scene_keys)

                
                split_idx = int(len(scene_keys)*(train_split + 0.5*val_split))

                train_keys += scene_keys[:split_idx]
                val_keys += scene_keys[split_idx:]

                print('Train Perc: ', len(train_keys) / (len(train_keys) + len(val_keys)))
            
            val_keys = ['data_13:02:17', 'data_19:58:40', 'data_15:50:55', 'data_16:22:44', 'data_15:40:22', 'data_17:08:14', 'data_15:37:13', 'data_18:38:30', 'data_13:56:07', 'data_15:22:59', 'data_13:33:54', 'data_13:18:11', 'data_19:36:17', 'data_14:38:16', 'data_13:04:13', 'data_12:04:43', 'data_16:37:57', 'data_15:38:38', 'data_16:40:44', 'data_17:59:00', 'data_20:57:07', 'data_16:03:52', 'data_16:40:36', 'data_19:31:51', 'data_16:45:24', 'data_21:09:57', 'data_17:26:17', 'data_15:01:27', 'data_14:02:16', 'data_13:29:09', 'data_14:22:29', 'data_16:43:00', 'data_13:46:04', 'data_15:13:04', 'data_16:45:58', 'data_13:33:29', 'data_17:17:50', 'data_11:19:28', 'data_17:45:27', 'data_16:00:55', 'data_15:03:19', 'data_16:06:05', 'data_16:02:46', 'data_17:41:00', 'data_17:35:45', 'data_14:05:06', 'data_18:22:47', 'data_17:02:46', 'data_15:08:23', 'data_16:15:15', 'data_19:00:23', 'data_11:50:57', 'data_15:19:33', 'data_14:52:27', 'data_16:58:53', 'data_11:44:50', 'data_16:10:21', 'data_13:10:05', 'data_17:48:24', 'data_18:09:10', 'data_18:01:35', 'data_13:34:59', 'data_12:48:23', 'data_22:17:48', 'data_16:57:05', 'data_16:49:20', 'data_17:51:34', 'data_12:54:21', 'data_16:23:48', 'data_14:24:32', 'data_16:18:35', 'data_14:26:22', 'data_16:11:06', 'data_11:58:17', 'data_17:13:00', 'data_19:34:02', 'data_13:29:42', 'data_17:20:01', 'data_15:20:09', 'data_16:53:34', 'data_15:25:56']
            
            print('Train Keys: ', len(train_keys))
            print('Validation Keys: ', len(val_keys))
            print('Validation Keys: ', val_keys)

        elif split_style == 'diversity_ablation':

            assert(diversity_scenes < len(self.scene_to_keys.keys()), "Error: number of train scenes for diversity ablations cannot be {}".format(len(self.scene_to_keys.keys())))

            ordered_scenes = []; ordered_trajs = []
            
            for scene, traj in self.scene_to_keys.items():

                ordered_scenes.append(scene)
                ordered_trajs.append(len(traj))
            

            ordered_index = sorted(range(0, len(ordered_trajs)), key = lambda x: ordered_trajs[x])

            ordered_trajs = list(sorted(ordered_trajs))
            ordered_scenes = [ordered_scenes[i] for i in ordered_index]

            print('EVAL SCENE: {} has {} trajectories'.format(ordered_scenes[-1], ordered_trajs[-1]))
            val_keys = self.scene_to_keys[ordered_scenes[-1]]
            other_scenes = list(reversed(ordered_scenes[:-1]))
            other_trajs = list(reversed(ordered_trajs[:-1]))


            num_per_scene = int(max_trajectories/diversity_scenes)
            train_keys = []

            for i in range(diversity_scenes):
                train_keys += random.sample(self.scene_to_keys[other_scenes[i]], num_per_scene)
            
            if len(train_keys) < max_trajectories:

                random_scene = random.sample(other_scenes[:diversity_scenes], 1)[0]
                train_keys += random.sample(self.scene_to_keys[random_scene], max_trajectories-len(train_keys))


        if 'attribute_limits.json' not in os.listdir('./lanmp_dataloader'):
            body_pose_lim, body_orientation_lim, end_effector_pose_lim = self.determine_min_max_range([train_keys, val_keys, test_keys])
        else:

            with open('./lanmp_dataloader/attribute_limits.json') as f:
                attribute_limits = json.load(f)
            body_pose_lim, body_orientation_lim, end_effector_pose_lim = attribute_limits[0], attribute_limits[1], attribute_limits[2]

        self.train_dataset = RT1Dataset(train_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)
        self.val_dataset = RT1Dataset(val_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)
        # self.test_dataset = RT1Dataset(test_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)

    def determine_min_max_range(self, data_subset_keys):

        body_pose = {'min_x': float('inf'), 'max_x': float('-inf'), 'min_y': float('inf'), 'max_y': float('-inf'), 'min_z': float('inf'), 'max_z':float('-inf')}
        body_orientation = {'min_yaw': float('inf'), 'max_yaw': float('-inf')}
        end_effector_pose = {'min_x': float('inf'), 'max_x': float('-inf'), 'min_y': float('inf'), 'max_y': float('-inf'), 'min_z': float('inf'), 'max_z': float('-inf')}
        
        


        with h5py.File(DATASET_PATH, 'r') as hdf:
            for dataset_keys in data_subset_keys:

                if dataset_keys is None:
                    continue


                for i in range(len(dataset_keys)):
                    prev_body_x = None
                    prev_body_y = None
                    prev_body_z = None
                    prev_body_yaw = None
                    prev_ee_x = None
                    prev_ee_y = None
                    prev_ee_z = None

                    print('Index: {} of {}'.format(i, len(dataset_keys)))
                    traj_group = hdf[dataset_keys[i]]
                    traj_steps = list(traj_group.keys())
                    traj_steps.sort(key=sort_folders) 

                    for j in range(len(traj_steps)):

                        step_metadata = json.loads(traj_group[traj_steps[j]].attrs['metadata'])

                        body_x = step_metadata['steps'][0]['state_body'][0]
                        body_y = step_metadata['steps'][0]['state_body'][1]
                        body_z = step_metadata['steps'][0]['state_body'][2]
                        
                        body_yaw = step_metadata['steps'][0]['state_body'][3]
                        

                        ee_x = step_metadata['steps'][0]['state_ee'][0]
                        ee_y = step_metadata['steps'][0]['state_ee'][1]
                        ee_z = step_metadata['steps'][0]['state_ee'][2]



                        body_pose['min_x'] = min(body_pose['min_x'], body_x - prev_body_x if prev_body_x is not None else 0)
                        body_pose['max_x'] = max(body_pose['max_x'], body_x - prev_body_x if prev_body_x is not None else 0)

                        body_pose['min_y'] = min(body_pose['min_y'], body_y - prev_body_y if prev_body_y is not None else 0)
                        body_pose['max_y'] = max(body_pose['max_y'], body_y - prev_body_y if prev_body_y is not None else 0)
                        
                        body_pose['min_z'] = min(body_pose['min_z'], body_z - prev_body_z if prev_body_z is not None else 0)
                        body_pose['max_z'] = max(body_pose['max_z'], body_z - prev_body_z if prev_body_z is not None else 0)

                        body_orientation['min_yaw'] = min(body_orientation['min_yaw'], body_yaw - prev_body_yaw if prev_body_yaw is not None else 0)
                        body_orientation['max_yaw'] = max(body_orientation['max_yaw'], body_yaw - prev_body_yaw if prev_body_yaw is not None else 0)

                        end_effector_pose['min_x'] = min(end_effector_pose['min_x'], ee_x - prev_ee_x if prev_ee_x is not None else 0)
                        end_effector_pose['max_x'] = max(end_effector_pose['max_x'], ee_x - prev_ee_x if prev_ee_x is not None else 0)

                        end_effector_pose['min_y'] = min(end_effector_pose['min_y'], ee_y - prev_ee_y if prev_ee_y is not None else 0)
                        end_effector_pose['max_y'] = max(end_effector_pose['max_y'], ee_y - prev_ee_y if prev_ee_y is not None else 0)

                        end_effector_pose['min_z'] = min(end_effector_pose['min_z'], ee_z - prev_ee_z if prev_ee_z is not None else 0)
                        end_effector_pose['max_z'] = max(end_effector_pose['max_z'], ee_z - prev_ee_z if prev_ee_z is not None else 0)


                        prev_body_x = body_x
                        prev_body_y = body_y
                        prev_body_z = body_z
                        prev_body_yaw = body_yaw
                        prev_ee_x = ee_x
                        prev_ee_y = ee_y
                        prev_ee_z = ee_z 

        

        #cache the saved max and min values if already computed to save time
        attribute_limits = [body_pose, body_orientation, end_effector_pose]
        with open('./lanmp_dataloader/attribute_limits.json', 'w') as f:
            json.dump(attribute_limits, f)
        
        
        return body_pose, body_orientation, end_effector_pose

    def collate_batches(self, batch, shuffle_batch = False):
        
        
        collated_batch = []

        # merging batch elements with variable length
        for out in range(len(batch[0])):
            collated_output = []
            for idx in range(len(batch)):
                if batch[idx][out].dtype.type == np.str_:
                    collated_output.append(batch[idx][out])
                else:
                    collated_output.append(torch.from_numpy(batch[idx][out]))
            
            if batch[idx][out].dtype.type!=np.str_:
                collated_output = torch.cat(collated_output, dim=0)
            else:
                
                collated_output = np.concatenate(collated_output, axis=0)
            
            collated_batch.append(collated_output)

        #shuffling all the batched samples across the trajectories to get random order
        if shuffle_batch:
            permutation = torch.randperm(collated_batch[0].size(0))
            
            for i in range(len(collated_batch)):
                collated_batch[i] = collated_batch[i][permutation]
        
        return collated_batch
                    








class RT1Dataset(Dataset):

    

    def __init__(self, data_split_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim, tokenize_action=True):

        #stores the keys in the dataset for the appropriate split (train, validation or test)
        self.dataset_keys = data_split_keys
        self.body_pose_lim = body_pose_lim
        self.body_orientation_lim = body_orientation_lim
        self.end_effector_pose_lim = end_effector_pose_lim
        self.num_bins = 254

        self.tokenize_action = tokenize_action

        self.hdf =  h5py.File(DATASET_PATH, 'r')
    
    def __len__(self):
        return len(self.dataset_keys)


    def make_data_discrete(self, dictionary):

        
           
        #body x, y, z coordinate
        dictionary['body_position_deltas'][:,0] = 1 + (dictionary['body_position_deltas'][:,0] - self.body_pose_lim['min_x'])/ (self.body_pose_lim['max_x'] - self.body_pose_lim['min_x'] ) * self.num_bins
        dictionary['body_position_deltas'][:,0] = dictionary['body_position_deltas'][:,0].astype(int)
        
        if self.body_pose_lim['max_y'] - self.body_pose_lim['min_y'] > 0:
            dictionary['body_position_deltas'][:,1] = 1 + (dictionary['body_position_deltas'][:,1] - self.body_pose_lim['min_y'])/(self.body_pose_lim['max_y'] - self.body_pose_lim['min_y'] ) * self.num_bins  
        else:
            dictionary['body_position_deltas'][:,1].fill(0)
        dictionary['body_position_deltas'][:,1] = dictionary['body_position_deltas'][:,1].astype(int)
        
        dictionary['body_position_deltas'][:,2] = 1 + (dictionary['body_position_deltas'][:,2] - self.body_pose_lim['min_z'])/ (self.body_pose_lim['max_z'] - self.body_pose_lim['min_z'] ) * self.num_bins
        dictionary['body_position_deltas'][:,2] = dictionary['body_position_deltas'][:,2].astype(int)

        #body yaw and pitch
        dictionary['body_yaw_deltas'] = 1 + (dictionary['body_yaw_deltas'] - self.body_orientation_lim['min_yaw']) / (self.body_orientation_lim['max_yaw'] - self.body_orientation_lim['min_yaw']) * self.num_bins
        dictionary['body_yaw_deltas'] = dictionary['body_yaw_deltas'].astype(int)

        #end effector x, y, z coordinate
        dictionary['arm_position_deltas'][:,0] = 1 + (dictionary['arm_position_deltas'][:,0] - self.end_effector_pose_lim['min_x'])/ (self.end_effector_pose_lim['max_x'] - self.end_effector_pose_lim['min_x'] ) * self.num_bins
        dictionary['arm_position_deltas'][:,0] = dictionary['arm_position_deltas'][:,0].astype(int)

        dictionary['arm_position_deltas'][:,1] = 1 + (dictionary['arm_position_deltas'][:,1] - self.end_effector_pose_lim['min_y'])/ (self.end_effector_pose_lim['max_y'] - self.end_effector_pose_lim['min_y'] ) * self.num_bins
        dictionary['arm_position_deltas'][:,1] = dictionary['arm_position_deltas'][:,1].astype(int)

        dictionary['arm_position_deltas'][:,2] = 1 + (dictionary['arm_position_deltas'][:,2] - self.end_effector_pose_lim['min_z'])/ (self.end_effector_pose_lim['max_z'] - self.end_effector_pose_lim['min_z'] ) * self.num_bins
        dictionary['arm_position_deltas'][:,2] =  dictionary['arm_position_deltas'][:,2].astype(int)

        #find if and where episode terminates so you can fill those entries with 0s
        if 1.0 in dictionary['terminate_episode']:
            terminate_idx = np.where(np.array(dictionary['terminate_episode'])>0)[0][0]

            dictionary['body_position_deltas'][terminate_idx:,:].fill(0)
            dictionary['body_yaw_deltas'][terminate_idx:].fill(0)
            dictionary['arm_position_deltas'][terminate_idx:,:].fill(0)

        
        return dictionary
           
    
    def detokenize_continuous_data(self, dictionary):

        if dictionary['curr_mode'] == 'stop':
            dictionary['body_position_delta'] = [[0.0, 0.0, 0.0]]
            dictionary['body_yaw_delta'] = [[0.0]]
            dictionary['arm_position_deltas'] = [[0.0, 0.0, 0.0]]

        else:
            dictionary['body_position_delta'][0][0] = (dictionary['body_position_delta'][0][0] - 1) * (self.body_pose_lim['max_x'] - self.body_pose_lim['min_x']) / self.num_bins + self.body_pose_lim['min_x']
            dictionary['body_position_delta'][0][1] = (dictionary['body_position_delta'][0][1] - 1) * (self.body_pose_lim['max_y'] - self.body_pose_lim['min_y']) / self.num_bins + self.body_pose_lim['min_y']
            dictionary['body_position_delta'][0][2] = (dictionary['body_position_delta'][0][2] - 1) * (self.body_pose_lim['max_z'] - self.body_pose_lim['min_z']) / self.num_bins + self.body_pose_lim['min_z']

            dictionary['body_yaw_delta'][0][0] = (dictionary['body_yaw_delta'][0][0] - 1) * (self.body_orientation_lim['max_yaw'] - self.body_orientation_lim['min_yaw']) / self.num_bins + self.body_orientation_lim['min_yaw']


            dictionary['arm_position_delta'][0][0] = (dictionary['arm_position_delta'][0][0] - 1) * (self.end_effector_pose_lim['max_x'] - self.end_effector_pose_lim['min_x']) / self.num_bins + self.end_effector_pose_lim['min_x']
            dictionary['arm_position_delta'][0][1] = (dictionary['arm_position_delta'][0][1] - 1) * (self.end_effector_pose_lim['max_y'] - self.end_effector_pose_lim['min_y']) / self.num_bins + self.end_effector_pose_lim['min_y']
            dictionary['arm_position_delta'][0][2] = (dictionary['arm_position_delta'][0][2] - 1) * (self.end_effector_pose_lim['max_z'] - self.end_effector_pose_lim['min_z']) / self.num_bins + self.end_effector_pose_lim['min_z']
        return dictionary


    def make_data_discrete_old(self, dictionary):

        if not bool(dictionary['is_terminal']):
        
            #body x, y, z coordinate
            dictionary['body_position'][0] = 1 + int( (dictionary['body_position'][0] - self.body_pose_lim['min_x'])/ (self.body_pose_lim['max_x'] - self.body_pose_lim['min_x'] ) * self.num_bins)
            
            dictionary['body_position'][1] = 1 + int( (dictionary['body_position'][1] - self.body_pose_lim['min_y'])/ (self.body_pose_lim['max_y'] - self.body_pose_lim['min_y'] ) * self.num_bins) if self.body_pose_lim['max_y'] - self.body_pose_lim['min_y'] > 0 else 0
            
            dictionary['body_position'][2] = 1 + int( (dictionary['body_position'][2] - self.body_pose_lim['min_z'])/ (self.body_pose_lim['max_z'] - self.body_pose_lim['min_z'] ) * self.num_bins)

            #body yaw and pitch
            dictionary['body_yaw'] = 1 + int( (dictionary['body_yaw'] - self.body_orientation_lim['min_yaw']) / (self.body_orientation_lim['max_yaw'] - self.body_orientation_lim['min_yaw']) * self.num_bins)

            #end effector x, y, z coordinate
            dictionary['arm_position'][0] = 1 + int( (dictionary['arm_position'][0] - self.end_effector_pose_lim['min_x'])/ (self.end_effector_pose_lim['max_x'] - self.end_effector_pose_lim['min_x'] ) * self.num_bins)
            dictionary['arm_position'][1] = 1 + int( (dictionary['arm_position'][1] - self.end_effector_pose_lim['min_y'])/ (self.end_effector_pose_lim['max_y'] - self.end_effector_pose_lim['min_y'] ) * self.num_bins)
            dictionary['arm_position'][2] = 1 + int( (dictionary['arm_position'][2] - self.end_effector_pose_lim['min_z'])/ (self.end_effector_pose_lim['max_z'] - self.end_effector_pose_lim['min_z'] ) * self.num_bins)
       
        #if terminal action is chosen, then produce 'no action' discrete value for each of the state variables
        else:
            dictionary['body_position'][0] = 0
            dictionary['body_position'][1] = 0
            dictionary['body_position'][2] = 0

            dictionary['body_yaw'] = 0

            dictionary['arm_position'][0] = 0
            dictionary['arm_position'][1] = 0
            dictionary['arm_position'][2] = 0
        
    def get_head_pitch(self, action):

        value = 0

        if action == 'LookDown':
            value = 1
        elif action == 'LookUp':
            value = 2

        return value
    
    def detokenize_head_pitch(self, token):

        tokenization_dict = {0:None, 1:'LookDown', 2:'LookUp'}

        return tokenization_dict[token]

    def get_mode(self, action):

        #mode: (0) stop, (1) body, (2) yaw,  (3) manipulation, (4) grasping, (5) head pitch
        
        value = None

        if action == 'stop':
            value = 0
        elif action in set( ['LookDown', 'LookUp']):
            value = 5
        elif action in set(['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']):
            value = 1
        elif action in set(['PickupObject', 'ReleaseObject']):
            value = 4
        elif action in set(['MoveArm', 'MoveArmBase']):
            value = 3
        elif action  == 'RotateAgent':
            value = 2
        
        assert(type(value)==int, 'Get Mode didn\'t return an int')
        return value
    
    def detokenize_mode(self, token):

        tokenization_dict = {0: 'stop', 1:'MoveAgent', 2:'RotateAgent', 3:'MoveArm', 4:'PickupReleaseObject', 5:'PitchAgent'}

        return tokenization_dict[token]
    
    def detokenize_action(self, detokenized_mode, body_position_delta, body_yaw_delta, arm_position_delta, detokenized_pickup_release, detokenized_head_pitch):

        
        if detokenized_mode == 'PickupReleaseObject':
            return detokenized_pickup_release

        elif detokenized_mode == 'PitchAgent':
            return detokenized_head_pitch
        else:
            return detokenized_mode


    def get_pickup_release(self, action):

        if action == 'PickupObject':
            value = 1
        elif action == 'ReleaseObject':
            value = 2
        else: 
            value = 0
        
        return value

    def detokenize_pickup_release(self, token):

        tokenization_dict = {0:None, 1:'PickupObject', 2:'ReleaseObject'}
        return tokenization_dict[token]
    
    def __getitem__(self, idx):
        
        # pdb.set_trace()
        
        traj_group = self.hdf[self.dataset_keys[idx]]
        
        traj_steps = list(traj_group.keys())
        traj_steps.sort(key=sort_folders) 

        #extract the NL command
        json_str = traj_group[traj_steps[0]].attrs['metadata']
        traj_json_dict = json.loads(json_str)
        nl_command = traj_json_dict['nl_command']


        #compute remainder in case padding of action tokens and observations needed
        padding_length = 6 - (len(traj_steps)%6) if len(traj_steps)%6 > 0 else 0
        terminate = False

        start = 0; end = min(len(traj_steps), 6)

        #return list of dictionaries with attributes required for RT1
        all_image_obs = []
        all_nl_commands = []
        all_is_terminal = []
        all_pickup_release = []
        all_body_position_deltas = []
        all_body_yaw_deltas = []
        all_body_pitches = []
        all_arm_position_deltas = []
        all_control_mode = []

        all_pad_lengths = []



        #build the dictionary for each sequence
        while end <= len(traj_steps) and not terminate:

            '''
                mode: stop, body, yaw, manipulation, grasping, head pitch
                gripper: (x, y, z, grasp)
                body: (x, y, yaw, look up/down)
            '''
            image_obs = []
            nl_commands = []
            body_position_deltas = []
            body_yaw_deltas = []
            arm_position_deltas = []
            terminate_episodes = []
            pickup_releases = []
            body_pitches = []
            control_modes = []

            for i in range(start, end):

                #visual observation
                ith_obs = np.array(traj_group[traj_steps[i]]['rgb_{}'.format(i)])
                image_obs.append(ith_obs)

                #natural language command
                nl_commands.append(nl_command)

                current_metadata = json.loads(traj_group[traj_steps[i]].attrs['metadata'])
                

                if i < len(traj_steps)-1:

                    next_metadata = json.loads(traj_group[traj_steps[i+1]].attrs['metadata'])
                
                    #body position, body yaw, arm position
                    body_position_delta = np.array(next_metadata['steps'][0]['state_body'][:3])-np.array(current_metadata['steps'][0]['state_body'][:3])
                    body_yaw_delta = next_metadata['steps'][0]['state_body'][3] - current_metadata['steps'][0]['state_body'][3]
                    arm_position_delta = np.array(next_metadata['steps'][0]['state_ee'][:3]) - np.array(current_metadata['steps'][0]['state_ee'][:3])

                    #terminate episode / pick up release / body pitch / mode
                    terminate_episode = int(i == len(traj_steps)-1)
                    pickup_release = self.get_pickup_release(next_metadata['steps'][0]['action'])
                    body_pitch = self.get_head_pitch(next_metadata['steps'][0]['action'])
                    control_mode = self.get_mode(next_metadata['steps'][0]['action'])
                else:

                    #body position, body yaw, arm positon -- for last step
                    body_position_delta = np.array([0.0, 0.0, 0.0])
                    body_yaw_delta = 0.0
                    arm_position_delta = np.array([0.0, 0.0, 0.0])

                    #is terminal / pick up release / body pitch / mode -- for last step
                    terminate_episode = int(i == len(traj_steps)-1)
                    pickup_release = self.get_pickup_release(None)
                    body_pitch = self.get_head_pitch(None)
                    control_mode = self.get_mode('stop')

                body_position_deltas.append(body_position_delta)
                body_yaw_deltas.append(body_yaw_delta)
                arm_position_deltas.append(arm_position_delta)
                terminate_episodes.append(terminate_episode)
                pickup_releases.append(pickup_release)
                body_pitches.append(body_pitch)
                control_modes.append(control_mode)

            

            #check for remainder and pad data with extra
            if end >= len(traj_steps) and padding_length > 0:
                
                for pad in range(0, padding_length):
                    
                    image_obs.append(ith_obs)
                    nl_commands.append(nl_command)

                    body_position_deltas.append(np.array([0.0, 0.0, 0.0]))
                    body_yaw_deltas.append(0.0)
                    arm_position_deltas.append(np.array([0.0, 0.0, 0.0]))
                    terminate_episodes.append(0)
                    pickup_releases.append(0.0)
                    body_pitches.append(0.0)
                    control_modes.append(0.0)
                
                terminate = True
            elif end >= len(traj_steps):
                terminate = True
                


            #pre-process and discretize numerical data 
            body_position_deltas = np.stack(body_position_deltas)
            body_yaw_deltas = np.stack(body_yaw_deltas)
            arm_position_deltas = np.stack(arm_position_deltas)
            
            if self.tokenize_action:
                
                tokenized_actions = {
                    'body_position_deltas': body_position_deltas,
                    'body_yaw_deltas': body_yaw_deltas,
                    'arm_position_deltas': arm_position_deltas,
                    'terminate_episode': terminate_episodes
                }
                
                tokenized_actions = self.make_data_discrete(tokenized_actions)
                
                body_position_deltas = tokenized_actions['body_position_deltas']
                
                body_yaw_deltas = np.expand_dims(tokenized_actions['body_yaw_deltas'], axis=1)
                
                arm_position_deltas = tokenized_actions['arm_position_deltas']
                

            

            all_image_obs.append(np.stack(image_obs))
            all_nl_commands.append(np.stack(nl_commands))
            all_is_terminal.append(np.stack(terminate_episodes))
            all_pickup_release.append(np.stack(pickup_releases))
            all_body_position_deltas.append(body_position_deltas)
            all_body_yaw_deltas.append(body_yaw_deltas)
            all_body_pitches.append(np.stack(body_pitches))
            all_arm_position_deltas.append(arm_position_deltas)
            all_control_mode.append(np.stack(control_modes))

            all_pad_lengths.append(0 if not end >= len(traj_steps) else padding_length)
            

            start += 6
            end = min(end + 6, len(traj_steps))


            
        
        return np.stack(all_image_obs), np.stack(all_nl_commands), np.stack(all_is_terminal), np.stack(all_pickup_release), np.stack(all_body_position_deltas), np.stack(all_body_yaw_deltas), np.stack(all_body_pitches), np.stack(all_arm_position_deltas), np.stack(all_control_mode), np.stack(all_pad_lengths)



    def __getitem_old__(self, idx):
        
        
        traj_group = self.hdf[self.dataset_keys[idx]]
        
        traj_steps = list(traj_group.keys())
        traj_steps.sort(key=sort_folders) 

        #extract the NL command
        json_str = traj_group[traj_steps[0]].attrs['metadata']
        traj_json_dict = json.loads(json_str)
        nl_command = traj_json_dict['nl_command']

        start = 0; end = min(len(traj_steps), 6)

        #return list of dictionaries with attributes required for RT1
        all_image_obs = []
        all_nl_commands = []
        all_is_terminal = []
        all_pickup_release = []
        all_body_position = []
        all_body_yaw = []
        all_body_pitch = []
        all_arm_position = []
        all_mode = []



        #build the dictionary for each sequence
        while end < len(traj_steps):

            '''
                mode: stop, body, yaw, manipulation, grasping, head pitch
                gripper: (x, y, z, grasp)
                body: (x, y, yaw, look up/down)
            '''
            image_obs = []

            for i in range(start, end):
                ith_obs = np.array(traj_group[traj_steps[i]]['rgb_{}'.format(i)])
                
                image_obs.append(ith_obs)
            
            image_obs = np.stack(image_obs)
            
            
            
            before_end_step_metadata = json.loads(traj_group[traj_steps[end-1]].attrs['metadata'])
            end_step_metadata = json.loads(traj_group[traj_steps[end]].attrs['metadata'])                



            dictionary = {
                'observation': image_obs,
                'nl_command': nl_command, #DONE
                'is_terminal': int(end_step_metadata['steps'][0]['action']=='stop'), #DONE
                'pickup_release': self.get_pickup_release(end_step_metadata['steps'][0]['action']), #DONE
                'body_position': np.array(end_step_metadata['steps'][0]['state_body'][:3])-np.array(before_end_step_metadata['steps'][0]['state_body'][:3]), #DONE 
                'body_yaw': end_step_metadata['steps'][0]['state_body'][3] - before_end_step_metadata['steps'][0]['state_body'][3], #DONE
                'body_pitch': self.get_head_pitch(end_step_metadata['steps'][0]['action']), #DONE
                'arm_position': np.array(end_step_metadata['steps'][0]['state_ee'][:3]) - np.array(before_end_step_metadata['steps'][0]['state_ee'][:3]), #DONE
                'mode': self.get_mode(end_step_metadata['steps'][0]['action']) #DONE
            }

            #pre-process the data dictonary
            if self.tokenize_action:
                self.make_data_discrete(dictionary)


            all_image_obs.append(dictionary['observation'])
            all_nl_commands.append(dictionary['nl_command'])
            all_is_terminal.append(dictionary['is_terminal'])
            all_pickup_release.append(dictionary['pickup_release'])
            all_body_position.append(dictionary['body_position'])
            all_body_yaw.append(dictionary['body_yaw'])
            all_body_pitch.append(dictionary['body_pitch'])
            all_arm_position.append(dictionary['arm_position'])
            all_mode.append(dictionary['mode'])
            

            start += 1
            end += 1

        #add the terminal 'stop' step
        all_image_obs.append(dictionary['observation'])
        all_nl_commands.append(dictionary['nl_command'])
        all_is_terminal.append(1)
        all_pickup_release.append(0)
        all_body_position.append([0,0,0])
        all_body_yaw.append(0)
        all_body_pitch.append(0)
        all_arm_position.append([0,0,0])
        all_mode.append(0)



        
       
        return np.stack(all_image_obs), np.stack(all_nl_commands), np.expand_dims(np.stack(all_is_terminal), axis=1), np.expand_dims(np.stack(all_pickup_release), axis=1), np.stack(all_body_position), np.expand_dims(np.stack(all_body_yaw),axis=1), np.expand_dims(np.stack(all_body_pitch), axis=1), np.stack(all_arm_position), np.expand_dims(np.stack(all_mode), axis=1)


if __name__ == '__main__':

   
    dataset_manager = DatasetManager(0, 0.8, 0.1, 0.1)

    dataloader = DataLoader(dataset_manager.train_dataset, batch_size=3,
                        shuffle=True, num_workers=0, collate_fn= dataset_manager.collate_batches)

    val_dataloader = DataLoader(dataset_manager.val_dataset, batch_size=2,
                        shuffle=True, num_workers=0, collate_fn= dataset_manager.collate_batches)

    
    
    for batch, sample_batch in enumerate(dataloader):
        
        # print('BATCH {}:'.format(batch))
        # print('Num Steps: {}'.format(sample_batch[0].shape[0]))
        print('Batch {}: '.format(batch), sample_batch[0].shape[0])
        
        
   

   