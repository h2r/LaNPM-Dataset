import os
import sys

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import h5py
from PIL import Image
from tqdm import tqdm
from models.utils.data_utils import split_data

#mainly for debugging
import matplotlib.pyplot as plt


DATASET_PATH = '/users/ajaafar/data/shared/lanmp/lanmp_dataset.hdf5'

'''
train_keys, val_keys, test_keys = split_data(self.args.data, splits['train'], splits['val'], splits['test'])
'''


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

            #extract the NL command
            json_str = traj_group[traj_steps[0]].attrs['metadata']
            traj_json_dict = json.loads(json_str)
            nl_command = traj_json_dict['nl_command']

            start = 0; end = min(len(traj_steps), 6)

            #return list of dictionaries with attributes required for RT1
            data_sequence = []

            #build the dictionary for each sequence
            while end < len(traj_steps):
                
                dictionary = {

                    'observation': #300x300x18 (3*6 images)
                    'language_command':
                    'output_action':
                }

                for i in range(start, end):
                    


                
                #insert the 'output_action' directly from the end index


                data_sequence.append(dictionary)

                start += 1
                end += 1


            
            # scene = traj_json_dict['scene']
            return data_sequence

