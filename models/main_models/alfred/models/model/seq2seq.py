import os
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange

class Module(nn.Module):

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0 #for lang
        if args.class_mode:
            self.action_pad = -1 # action index for classification
            self.grasp_drop_class_num = 3
            self.up_down_class_num = 3
            self.base_class_num = 5
            self.mode_class_num = 7
            self.bin_add = 1
        else:
            self.action_pad = -2.0 #regression
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), self.args.demb)

        if self.args.class_mode:
            self.emb_mode = nn.Embedding(self.mode_class_num, self.args.demb)
            self.emb_base = nn.Embedding(self.base_class_num, self.args.demb)  # For base word action
            self.emb_yaw = nn.Embedding(self.args.bins+self.bin_add, self.args.demb) # For yaw of the base
            self.emb_eff_xyz = nn.Embedding(self.args.bins+self.bin_add, self.args.demb)  # For x, y, z end-effector position
            self.emb_grasp_drop = nn.Embedding(self.grasp_drop_class_num, self.args.demb)
            self.emb_up_down = nn.Embedding(self.up_down_class_num, self.args.demb)
            self.emb_action = {"emb_mode": self.emb_mode, "emb_base": self.emb_base, "emb_yaw": self.emb_yaw, "emb_eff_xyz": self.emb_eff_xyz, "emb_grasp_drop": self.emb_grasp_drop, "emb_up_down": self.emb_up_down}

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)

        # summary self.writer
        self.summary_writer = None

    def run_train(self, args=None, optimizer=None):
        '''
        training loop
        '''

        # args
        args = args or self.args

        # splits
        with open(self.args.split_keys, 'r') as f:
            split_keys = json.load(f)
            train = split_keys['train']
            test = split_keys['test']

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            index_to_keep = int(len(train) * self.args.dataset_fraction)
            train = train[:index_to_keep]
            index_to_keep = int(len(valid_seen) * self.args.dataset_fraction)
            valid_seen = valid_seen[:index_to_keep]
            index_to_keep = int(len(valid_unseen) * self.args.dataset_fraction)
            valid_unseen = valid_unseen[:index_to_keep]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[:16]
            test = test[:16]

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # dump config
        fconfig = os.path.join(self.args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(self.args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=self.args.lr)

        # display dout
        print("Saving to: %s" % self.args.dout)
        best_loss = {'train': 1e10, 'test': 1e10}
        train_iter, test_iter = 0, 0,
        total_train_results = list()
        total_test_results = list()
        for epoch in trange(0, self.args.epoch, desc='epoch'):
            print(f'train_iter: {train_iter}\n')
            m_train = collections.defaultdict(list) #dict where values are lists
            self.train() #puts model in training mode
            self.adjust_lr(optimizer, self.args.lr, epoch, decay_epoch=self.args.decay_epoch)
            random.shuffle(train) # shuffle every epoch
            epoch_losses = []
            for batch, feat in self.iterate(train, self.args.batch):
                out = self.forward(feat)
                loss = self.compute_loss(out, batch, feat) #loss for the whole batch

                # optimizer backward pass
                optimizer.zero_grad()
                scalar_loss = loss['action_low']
                std = loss['action_low_std']

                scalar_loss.backward() # performs gradients
                optimizer.step() # makes the change based on the gradients

                self.summary_writer.add_scalar('train/loss', scalar_loss, train_iter)
                scalar_loss = scalar_loss.detach().cpu()
                std = std.detach().cpu()
                epoch_losses.append({"mean_batch_loss_train": float(scalar_loss), "std_batch_loss_train": float(std)})
                train_iter += 1
            total_train_results.append(epoch_losses)

            # compute metrics for test
            avg_test_loss, avg_test_std = self.run_pred(test, args=self.args, name='test', iter=test_iter)
            total_test_results.append({"mean_loss_test": avg_test_loss, "std_loss_test": avg_test_std})

            stats = {'epoch': epoch, 'test': avg_test_loss}

            # new best test loss
            if avg_test_loss < best_loss['test']:
                print('\nFound new best test!! Saving...')

                fsave = os.path.join(self.args.dout, 'best_test.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_test.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(self.args.dout, 'test.debug.preds.json')
                best_loss['test'] = avg_test_loss

            # save the latest checkpoint
            if self.args.save_every_epoch:
                fsave = os.path.join(self.args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(self.args.dout, 'latest.pth')
            torch.save({
                'metric': stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
            pprint.pprint(stats)
            
            loss_stds_dict = {"train_results": total_train_results, "test_results": total_test_results}
            losses_stds_path = os.path.join(self.args.dout, f'losses_stds.json') 
            with open(losses_stds_path, "w") as file: 
                json.dump(loss_stds_dict, file)


    def run_pred(self, dev, args=None, name='dev', iter=0):
        '''
        test loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        # p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        sum_batch_losses = 0
        sum_batch_stds = 0
        sum_feats = 0
        for batch, feat in self.iterate(dev, self.args.batch):
            out = self.forward(feat)
            loss = self.compute_loss(out, batch, feat) #avg loss for whole batch

            sum_loss = float(loss['action_low'].detach().cpu()) * len(feat) # get back the loss sum from the avg
            sum_batch_losses += sum_loss
            sum_feats += len(feat)

            sum_std = np.square(float(loss['action_low_std'].detach().cpu()))*len(feat)
            sum_batch_stds += sum_std

            self.summary_writer.add_scalar("%s/loss" % (name), sum_loss, dev_iter)
            dev_iter += 1

        avg_loss = sum_batch_losses/sum_feats
        avg_std = np.sqrt(sum_batch_stds/sum_feats)
        return avg_loss, avg_std

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['repeat_idx']))

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[i]['action_low'].split(),
            }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.pp_data, task, '%s' % self.args.pp_folder, 'ann_0.json')
        with open(json_path) as f:
            #check the split returned here. says 'train', but may not be a problem
            data = json.load(f)
        return data

    
    def iterate(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        for i in trange(0, len(data), batch_size, desc='batch'):
            tasks = data[i:i+batch_size]
            batch = [self.load_task_json(task) for task in tasks]
            feat = self.featurize(batch)
            yield batch, feat

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave)
        model = cls(save['args'], save['vocab'])
        model.load_state_dict(save['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save['optim'])
        return model, optimizer

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True