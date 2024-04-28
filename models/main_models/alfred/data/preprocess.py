import os
import json
import revtok
import torch
import copy
import progressbar
from vocab import Vocab
from model.seq2seq import Module as model
from gen.utils.py_util import remove_spaces_and_lower
from models.utils.data_utils import split_data
import h5py
import numpy as np

class Dataset(object):

    def __init__(self, args, vocab=None):
        self.args = args
        self.dataset_path = args.data
        self.pframe = args.pframe

        if vocab is None:
            self.vocab = {
                'word': Vocab(['<<pad>>', '<<seg>>', '<<goal>>']),
                # 'action_low': Vocab(['<<pad>>', '<<seg>>', '<<stop>>']),
                # 'action_high': Vocab(['<<pad>>', '<<seg>>', '<<stop>>']),
            }
        else:
            self.vocab = vocab

        self.word_seg = self.vocab['word'].word2index('<<seg>>', train=False)
        self.max_vals = [-1000000] * 10
        self.min_vals = [1000000] * 10
        self.bins = {}

    def find_all_max_min(self, split_keys_dict): #only called if class_mode is true
        for name, split_keys in split_keys_dict.items():
            #debugging
            if self.args.fast_epoch:
                split_keys = split_keys[:10]

            for task in split_keys: #task is a trajectory
                self.process_actions(task, None)

    @staticmethod
    def numericalize(vocab, words, train=True):
        '''
        converts words to unique integers
        '''
        return vocab.word2index([w.strip().lower() for w in words], train=train)


    def preprocess_splits(self, splits):
        '''
        saves preprocessed data as jsons in specified folder
        '''
        train_keys, val_keys, test_keys = split_data(self.args.data, splits['train'], splits['val'], splits['test'])
        split_keys_dict = {'train':train_keys, 'val':val_keys, 'test':test_keys}
        #make this path reltive later
        with open("/users/ajaafar/data/ajaafar/NPM-Dataset/models/main_models/alfred/data/splits/split_keys_discrete.json", 'w') as f:
            json.dump(split_keys_dict, f)

        #double check this part
        self.find_all_max_min(split_keys_dict)
        if self.args.class_mode:
            self.discretize_actions()

        for name, split_keys in split_keys_dict.items():
            #debugging
            if self.args.fast_epoch:
                split_keys = split_keys[:10]

            for task in progressbar.progressbar(split_keys): #task is a trajectory
                with h5py.File(self.args.data, 'r') as hdf:
                    traj_group = hdf[task]
                    traj_steps = list(traj_group.keys())
                    first_step_key = traj_steps[0]  # Get the first step key
                    json_str = traj_group[first_step_key].attrs['metadata']
                    traj_json_dict = json.loads(json_str)
                    nl_command = traj_json_dict['nl_command']

                traj = {}
                # root & split
                traj['root'] = os.path.join(self.args.pp_data, task)
                traj['split'] = name

                # numericalize language
                self.process_language(traj, nl_command)

                # numericalize actions for train/valid splits
                # if 'test' not in name: # expert actions are not available for the test set
                self.process_actions(task, traj)

                # check if preprocessing storage folder exists
                preprocessed_folder = os.path.join(self.args.pp_data, task, self.args.pp_folder)
                if not os.path.isdir(preprocessed_folder):
                    os.makedirs(preprocessed_folder)

                # save preprocessed json
                preprocessed_json_path = os.path.join(preprocessed_folder, "ann_0.json")
                with open(preprocessed_json_path, 'w') as f:
                    json.dump(traj, f, sort_keys=True, indent=4)

        # save vocab in dout path
        vocab_dout_path = os.path.join(self.args.dout, '%s.vocab' % self.args.pp_folder) #looks something like this: /path/to/output/pp.vocab. saved in exp folder
        torch.save(self.vocab, vocab_dout_path)

        # save vocab in data path
        vocab_data_path = os.path.join(self.args.pp_data, '%s.vocab' % self.args.pp_folder) # #looks something like this: /path/to/output/pp.vocab. saved in data folder
        torch.save(self.vocab, vocab_data_path)

        # save max and min values
        max_min = {'max': self.max_vals, 'min': self.min_vals}
        max_min_path = os.path.join(self.args.pp_data, 'max_min.json')
        with open(max_min_path, 'w') as f:
            json.dump(max_min, f, indent=4)


    def process_language(self, traj, nl_command):
        '''
        ex and traj are single trajectory that have about 3 'high_descs' which are the step by step instructions and 'task_desc' which is the final goal.
        ex and traj are the same with traj having a few more dic keys for some metadata
        '''
        # tokenize language
        traj['ann'] = {
            'goal': revtok.tokenize(remove_spaces_and_lower(nl_command)) + ['<<goal>>'],
        }

        # numericalize language
        traj['num'] = {}
        traj['num']['lang_goal'] = self.numericalize(self.vocab['word'], traj['ann']['goal'], train=True)

    def find_min_max(self, state_body, state_ee):
        actions = state_body + state_ee
        for i in range(len(actions)):
            if actions[i] > self.max_vals[i]:
                self.max_vals[i] = actions[i]
            if actions[i] < self.min_vals[i]:
                self.min_vals[i] = actions[i]
    
    def get_deltas(self, state_body, state_ee, next_state_body, next_state_ee):
        state_body_delta = np.subtract(next_state_body, state_body)
        state_ee_delta = np.subtract(next_state_ee, state_ee)
        return state_body_delta.tolist(), state_ee_delta.tolist()

    def process_actions(self, task, traj):
        if traj is not None:
            traj['num']['action_low'] = []
        with h5py.File(self.args.data, 'r') as hdf:
            traj_group = hdf[task]
            traj_steps = list(traj_group.keys())
            for i, step in enumerate(traj_steps):
                if i==0 and traj is not None:
                    print(len(traj_steps))
                    print(len(traj_steps))
                json_str = traj_group[step].attrs['metadata']
                traj_json_dict = json.loads(json_str)
                state_body = traj_json_dict['steps'][0]['state_body']
                state_ee = traj_json_dict['steps'][0]['state_ee']
                
                if self.args.relative:
                    if i != len(traj_steps)-1: #works for both regression and classification modes
                        next_step = traj_steps[i+1]
                        next_json_str = traj_group[next_step].attrs['metadata']
                        next_traj_json_dict = json.loads(next_json_str)
                        next_state_body = next_traj_json_dict['steps'][0]['state_body']
                        next_state_ee = next_traj_json_dict['steps'][0]['state_ee']
                        state_body, state_ee = self.get_deltas(state_body, state_ee, next_state_body, next_state_ee)

                if traj is None:
                    self.find_min_max(state_body, state_ee)
                
                if not self.args.class_mode: #regression
                    traj['num']['action_low'].append(
                        {'state_body': state_body, 'state_ee': state_ee}
                    )
                elif self.args.class_mode and traj is not None: #classification
                    def get_indices():
                        final_action_indices = []
                        state = state_body + state_ee
                        for i, val in enumerate(state):
                            bin_indx = np.digitize(val, self.bins[i])
                            #in case it thinks the value is equal to the max in the bins due to floating point precision error
                            if bin_indx == len(self.bins[i]):
                                bin_indx = len(self.bins[i])-1
                            final_action_indices.append(int(bin_indx-1)) #subtract since digitize returns 1-based indexing
                        traj['num']['action_low'].append(
                            {'state_body': final_action_indices[:4], 'state_ee': final_action_indices[4:]}
                        )
                    # add action indices, but skip the last action for relative actions since they use deltas
                    if self.args.relative:
                        if i != len(traj_steps)-1:
                            get_indices()
                    else:
                        get_indices()        
                        
            if traj is not None:
                #append end/stop action index of bins
                traj['num']['action_low'].append(
                        {'state_body': [0]*4, 'state_ee': [0]*6}
                )

    def discretize_actions(self):
        # Discretize each dimension
        for dim in range(len(self.min_vals)):
            # Create bin edges using linspace
            bin_edges = np.linspace(self.min_vals[dim], self.max_vals[dim], self.args.bins + 1)
            bin_edges = np.insert(bin_edges, 0, -1000) #adding bin for pad action (index 1)
            bin_edges = np.insert(bin_edges, 0, -2000) #adding bin for stop action (index 0)
            # in the case where self.args.bins is 256, then there are now 258 bins and len(bin_edges) is 259. index for last bin is 257
            self.bins[dim] = bin_edges

        bins_path = os.path.join(self.args.pp_data, 'bins.json')
        with open(bins_path, 'w') as f:
            json.dump({key: value.tolist() for key, value in self.bins.items()}, f, indent=4)