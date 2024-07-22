import json
import pprint
import random
import time
import torch
import torch.multiprocessing as mp
from models.nn.resnet import Resnet
from data.preprocess import Dataset
from importlib import import_module
import os

class Eval(object):

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        with open(self.args.split_keys, 'r') as f:
            self.splits = json.load(f)

        # load model
        print("Loading: ", self.args.model_path)
        M = import_module(self.args.model)
        self.model, optimizer = M.Module.load(self.args.model_path)
        self.model.share_memory()
        self.model.eval()
        self.model.test_mode = True

        # updated args
        self.model.args.dout = self.args.model_path.replace(self.args.model_path.split('/')[-1], '')


        # load resnet
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

        # gpu
        if self.args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
    
        files = self.splits['test']

        # debugging: fast epoch
        if self.args.fast_epoch:
            files = files[:16]

        if self.args.shuffle:
            random.shuffle(files)
        for traj in files:
            task_queue.put(traj)
        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        self.run(self.model, self.resnet, task_queue, self.args, lock, self.successes, self.failures, self.results)
        # for n in range(self.args.num_threads):
            # thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
            #                                            self.successes, self.failures, self.results))
            # thread.start()
            # threads.append(thread)

        # for t in threads:
        #     t.join()

        # save
        # self.save_results()

    @classmethod
    def setup_scene(cls, env, traj_data, args):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_name = traj_data['scene']
        env.reset(scene_name)

        # print goal instr
        print("Task: %s" % (traj_data['ann']['task_desc']))

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures):
        raise NotImplementedError()

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()
