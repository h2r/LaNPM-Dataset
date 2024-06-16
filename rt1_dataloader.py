import os
import sys

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
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

DATASET_PATH = '/oscar/data/stellex/shared/lanmp/lanmp_dataset.hdf5'

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

def sort_folders(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

class DatasetManager(object):
    '''
    NOTE: kwargs should contain a dictionary with keys {'train_split' : x, 'val_split': y, 'test_split':z} where x+y+z = 1
    '''
    def __init__(self, **kwargs):
        train_keys, val_keys, test_keys = split_data(DATASET_PATH, kwargs['train_split'], kwargs['val_split'], kwargs['test_split'])

        self.train_dataloader = DataLoader(train_keys)
        self.val_dataloader = DataLoader(val_keys)
        self.test_dataloader = DataLoader(test_keys)


class DataLoader(object):

    

    def __init__(self, data_split_keys):

        #stores the keys in the dataset for the appropriate split (train, validation or test)
        self.dataset_keys = data_split_keys
    
    def __len__(self):

        if self.train:
            return len(self.dataset_keys)
    
    def __getitem__(self, idx):
        
        with h5py.File(DATASET_PATH, 'r') as hdf:
            traj_group = hdf[self.dataset_keys[idx]]
            
            traj_steps = list(traj_group.keys())
            traj_steps.sort(key=sort_folders) 

            #extract the NL command
            json_str = traj_group[traj_steps[0]].attrs['metadata']
            traj_json_dict = json.loads(json_str)
            nl_command = traj_json_dict['nl_command']

            start = 0; end = min(len(traj_steps), 6)

            #return list of dictionaries with attributes required for RT1
            data_sequence = []

            '''
            is_terminal
            12:07
            2. gripper openness (edited) 
            12:07
            3. base displacement (coordinate)
            12:08
            4. image observation
            '''

            #build the dictionary for each sequence
            while end <= len(traj_steps):
                
                # dictionary = {

                #     'observation': #300x300x18 (3*6 images)
                #     'language_command': nl_command
                #     'output_action':
                # }

                for i in range(start, end):
                    ith_obs = traj_group[traj_steps[i]]['rgb_{}'.format(i)]
                    

                end_step_metadata = json.loads(traj_group[traj_steps[end-1]].attrs['metadata'])


                next_step_metadata = json.loads(traj_group[traj_steps[end]].attrs['metadata']) if end < len(traj_steps) else end_step_metadata

                # dictionary = {
                #     # 'observation': ,
                #     'nl_command': nl_command,
                #     'is_terminal': end >= len(traj_steps),
                #     'gripper_openness': len(end_step_metadata['steps'][0]['held_objs'])==0,
                #     'body_position': end_step_metadata['steps'][0]['state_body'][:3],
                #     'arm_position': end_step_metadata['steps'][0]['state_ee'][:3],
                #     'body_orientation': end_step_metadata['steps'][0]['state_body'][3:],
                #     'arm_orientation':end_step_metadata['steps'][0]['state_ee'][3:],
                #     'mode': ,
                #     'label_gripper_openness': len(next_step_metadata['steps'][0]['held_objs'])==0,
                #     'label_body_position': next_step_metadata['steps'][0]['state_body'][:3],
                #     'label_arm_position': next_step_metadata['steps'][0]['state_ee'][:3],
                #     'label_body_orientation': next_step_metadata['steps'][0]['state_body'][3:],
                #     'label_arm_orientation': next_step_metadata['steps'][0]['state_ee'][3:],
                #     'label_mode':
                # }

                #insert the 'output_action' directly from the end index


                data_sequence.append(dictionary)

                start += 1
                end += 1


            
            # scene = traj_json_dict['scene']
            return data_sequence


if __name__ == '__main__':

    dataset_keys, __, __ = split_data(DATASET_PATH, 0.7, 0.2, 0.1)


    with h5py.File(DATASET_PATH, 'r') as hdf:

        pdb.set_trace()
        traj_group = hdf[dataset_keys[0]]
        
        traj_steps = list(traj_group.keys())
        traj_steps.sort(key=sort_folders)

        #extract the NL command
        json_str = traj_group[traj_steps[0]].attrs['metadata']
        traj_json_dict = json.loads(json_str)
        nl_command = traj_json_dict['nl_command']

        start = 0; end = min(len(traj_steps), 6)

        #return list of dictionaries with attributes required for RT1
        data_sequence = []


