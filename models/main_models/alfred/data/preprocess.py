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


class Dataset(object):

    def __init__(self, args, vocab=None):
        self.args = args
        self.dataset_path = args.data
        self.pframe = args.pframe

        if vocab is None:
            self.vocab = {
                'word': Vocab(['<<pad>>', '<<seg>>', '<<goal>>']),
                'action_low': Vocab(['<<pad>>', '<<seg>>', '<<stop>>']),
                # 'action_high': Vocab(['<<pad>>', '<<seg>>', '<<stop>>']),
            }
        else:
            self.vocab = vocab

        self.word_seg = self.vocab['word'].word2index('<<seg>>', train=False)


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
        with open("/users/ajaafar/data/ajaafar/NPM-Dataset/models/main_models/alfred/data/splits/split_keys.json", 'w') as f:
            json.dump(split_keys_dict, f)
        for name, split_keys in split_keys_dict.items():
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
                if 'test' not in name: # expert actions are not available for the test set
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

    def process_actions(self, task, traj):
        # deal with missing end high-level action
        # self.fix_missing_high_pddl_end_action(ex)

        # end action for low_actions
        # end_action = {
        #     'api_action': {'action': 'NoOp'},
        #     'discrete_action': {'action': '<<stop>>', 'args': {}},
            # 'high_idx': ex['plan']['high_pddl'][-1]['high_idx']
        # }

        # init action_low and action_high
        # num_hl_actions = len(ex['plan']['high_pddl'])
        # traj['num']['action_low'] = [list() for _ in range(num_hl_actions)]  # temporally aligned with HL actions
        # # traj['num']['action_high'] = []
        # low_to_high_idx = []

        traj['num']['action_low'] = []
        with h5py.File(self.args.data, 'r') as hdf:
            traj_group = hdf[task]
            traj_steps = list(traj_group.keys())
            for step in traj_steps:
                json_str = traj_group[step].attrs['metadata']
                traj_json_dict = json.loads(json_str)
                state_body = traj_json_dict['steps'][0]['state_body']
                state_ee = traj_json_dict['steps'][0]['state_ee']

                traj['num']['action_low'].append(
                    {'state_body': state_body, 'state_ee': state_ee}
                )
            
            #append action to signal end/stop
            traj['num']['action_low'].append(
                    {'state_body': [-1, -1, -1], 'state_ee': [-1,-1,-1,-1,-1,-1]}
            )
                # high-level action index (subgoals)
                # high_idx = a['high_idx']
                # low_to_high_idx.append(high_idx)

                # low-level action (API commands)
                # traj['num']['action_low'][high_idx].append({
                #     'high_idx': a['high_idx'],
                #     'action': self.vocab['action_low'].word2index(a['discrete_action']['action'], train=True),
                #     'action_high_args': a['discrete_action']['args'],
                # })


                # interaction validity
                # valid_interact = 1 if model.has_interaction(a['discrete_action']['action']) else 0
                # traj['num']['action_low'][high_idx][-1]['valid_interact'] = valid_interact

        # low to high idx
        # traj['num']['low_to_high_idx'] = low_to_high_idx

        # high-level actions
        # for a in ex['plan']['high_pddl']:
        #     traj['num']['action_high'].append({
        #         'high_idx': a['high_idx'],
        #         'action': self.vocab['action_high'].word2index(a['discrete_action']['action'], train=True),
        #         'action_high_args': self.numericalize(self.vocab['action_high'], a['discrete_action']['args']),
        #     })

    def fix_missing_high_pddl_end_action(self, ex):
        '''
        appends a terminal action to a sequence of high-level actions
        '''
        if ex['plan']['high_pddl'][-1]['planner_action']['action'] != 'End':
            ex['plan']['high_pddl'].append({
                'discrete_action': {'action': 'NoOp', 'args': []},
                'planner_action': {'value': 1, 'action': 'End'},
                'high_idx': len(ex['plan']['high_pddl'])
            })


    def merge_last_two_low_actions(self, conv):
        '''
        combines the last two action sequences into one sequence
        '''
        extra_seg = copy.deepcopy(conv['num']['action_low'][-2])
        for sub in extra_seg:
            sub['high_idx'] = conv['num']['action_low'][-3][0]['high_idx']
            conv['num']['action_low'][-3].append(sub)
        del conv['num']['action_low'][-2]
        conv['num']['action_low'][-1][0]['high_idx'] = len(conv['plan']['high_pddl']) - 1