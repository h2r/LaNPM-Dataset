import os
import json
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask


class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        # self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # get action max and mins
        max_min_path = os.path.join(self.args.pp_data, 'max_min.json')
        with open(max_min_path, 'r') as f:
            max_min_data = json.load(f)
        max_vals = max_min_data['max']
        min_vals = max_min_data['min']

        # model to be finetuned
        decoder = vnn.ConvFrameMaskDecoder
        self.dec = decoder(max_vals, min_vals, self.emb_action, self.args.bins, self.bin_add, args.class_mode, args.demb, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           adapter_dropout=args.adapter_dropout,
                           teacher_forcing=args.dec_teacher_forcing,
                           action_dims = args.action_dims,
                           mode_class_num = self.mode_class_num, 
                           base_class_num = self.base_class_num,
                           up_down_class_num = self.up_down_class_num,
                           grasp_drop_class_num = self.grasp_drop_class_num)
        self.load_pretrained_model(args.finetune)
        self.freeze()
        
        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        # self.max_subgoals = 25

        # reset model
        self.reset()

    def freeze(self):
        """
        The layers to be fine-tuned:
            emb_word.weight
            dec.adapter.weight
            dec.adapter.bias
            dec.actor.weight
            dec.actor.bias
        """

        for name, param in self.named_parameters():
            if 'actor' not in name and 'emb_word' not in name and 'adapter' not in name:
                param.requires_grad = False

    def load_pretrained_model(self, path):

         # You might want to filter out unnecessary keys
        model_dict = self.state_dict()

        # Load the pretrained state dict
        pretrained_dict = torch.load(path)

        # Load pretrained model state dictionary
        pretrained_state_dict = pretrained_dict['model']

        #don't want to use actor and emb_word pretrained params so filtering them out
        filtered_dict = {key: value for key, value in pretrained_state_dict.items()
                     if 'actor' not in key and 'emb_word' not in key}
       
        #loads in the keys shared between the current model and the pretrained model while also removing the keys we want to fine-tune
        self.load_state_dict(filtered_dict, strict=False)

    def featurize(self, batch, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list) # for all trajs in the batch

        for ex in batch:
            #########
            # inputs
            #########

            # goal and instr language
            lang_goal = ex['num']['lang_goal']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal

            # append goal + instr
            lang_goal_instr = lang_goal
            feat['lang_goal_instr'].append(lang_goal_instr)

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = ex['root']
                root = 'data/feats' + root[34:] #delete later maybe
                im = torch.load(os.path.join(root, 'pp', self.feat_pt))

                
                num_low_actions =  len(ex['num']['action_low']) #already has the stop action so len is already +1
                if not self.args.relative:
                    im = torch.cat((im, im[-1].unsqueeze(0)), dim=0) #add one more frame that's a copy of the last frame so len(frames) matches len(actions) due to a stop action being added
                num_feat_frames = im.shape[0]

                # Modeling Quickstart (without filler frames)
                if num_low_actions == num_feat_frames:
                    feat['frames'].append(im)

                # Full Dataset (contains filler frames)
                #won't run for ours since every frame is accompanied by an action
                else:
                    keep = [None] * num_low_actions
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep[-1] = im[-1]  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append(ex['num']['action_low']) #append trajectory's sequence of actions. feat['action_low'] should end up being a list that's batch num long

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            else:
                # default: tensorize and pad sequence
                seqs = [vv.clone().detach().to(device=device, dtype=torch.float) if 'frames' in k else 
                                [{key: torch.tensor(value, device=device, dtype=torch.int) for key, value in d.items()} for d in vv] 
                                for vv in v]
                if k in {'action_low'}:
                #seqs is list of length batch where each item is a list that's traj length of dictionaries that contain actions where the actions are float tensors

                    # Determine the maximum length of any list in the seqs
                    max_length = max(len(lst) for lst in seqs)

                    template_dict = {
                        'mode': torch.tensor(self.action_pad),
                        'base_action': torch.tensor(self.action_pad),
                        'state_rot': torch.tensor(self.action_pad),
                        'state_ee': torch.full((3,), self.action_pad),
                        'grasp_drop': torch.tensor(self.action_pad),
                        'up_down': torch.tensor(self.action_pad)
                    }

                    # Pad each list in seqs to the maximum length
                    pad_seq = [
                        lst + [template_dict.copy() for _ in range(max_length - len(lst))] for lst in seqs
                    ]
                else:
                    pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
            
                feat[k] = pad_seq
        return feat


    def forward(self, feat, max_decode=300):
        cont_lang, enc_lang = self.encode_lang(feat)
        state_0 = cont_lang, torch.zeros_like(cont_lang) #self-attention encoding & 0-tensor with same len for every traj in the batch as an init hidden decoding
        frames = self.vis_dropout(feat['frames'])
        res = self.dec(enc_lang, frames, max_decode=max_decode, gold=feat['action_low'], state_0=state_0)
        feat.update(res)
        return feat


    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang_goal_instr = feat['lang_goal_instr']
        self.lang_dropout(emb_lang_goal_instr.data)
        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr) #LSTM encoding
        enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr) #self-attention encoding

        return cont_lang_goal_instr, enc_lang_goal_instr


    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }

    def step(self, feat, t, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)
            self.r_state['state_t'] = self.r_state['cont_lang'], torch.zeros_like(self.r_state['cont_lang'])

        # previous action embedding
        # may need to get rid of this embed
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        flag_reset = False 
        if t == 0:
            flag_reset = True
        action_logits, state_t, *_ = self.dec.step(self.r_state['enc_lang'], feat['frames'][:, 0], e_t=e_t, state_tm1=self.r_state['state_t'], flag_reset=flag_reset)
        
        # save states
        self.r_state['state_t'] = state_t

        max_action = self.get_max_action(action_logits)
        self.r_state['e_t'] = self.embed_action(max_action)

        # output formatting
        feat['out_action_low'] = action_logits.unsqueeze(0)
        return feat

    def get_max_action(self, action_logits):
        logits_1d = action_logits[:, :(1*self.mode_class_num)].view(-1, 1, self.mode_class_num)
        last = (1*self.mode_class_num)
        logits_base_1d = action_logits[:, last: last+(1*self.base_class_num)].view(-1, 1, self.base_class_num)
        last = last + (1*self.base_class_num)
        logits_yaw_1d = action_logits[:, last: last+(1*self.dec.num_bins)].view(-1, 1, self.dec.num_bins)
        last = last + (1*self.dec.num_bins)
        logits_3d = action_logits[:, last : last+(3 * self.dec.num_bins)].view(-1, 3, self.dec.num_bins)
        last = last + (3 * self.dec.num_bins)
        logits_2d = action_logits[:, last :  last + (2 * self.grasp_drop_class_num)].view(-1, 2, self.grasp_drop_class_num)
        max_1d = logits_1d.max(dim=2)[1]
        max_1d_base = logits_base_1d.max(dim=2)[1]
        max_1d_yaw = logits_yaw_1d.max(dim=2)[1]
        max_3d = logits_3d.max(dim=2)[1]
        max_2d = logits_2d.max(dim=2)[1]
        max_action = torch.cat((max_1d, max_1d_base, max_1d_yaw, max_3d, max_2d), dim=1)

        return max_action

    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing for real-time inference
        '''
        pred = {}
        for idx, ex in enumerate(batch):
            current_action_low = feat['out_action_low'][idx]
            alow = self.get_max_action(current_action_low)
            # remove padding tokens in case they're there (shouldn't be)
            if self.action_pad in alow:
                pad_start_idx = alow.index(self.action_pad)
                alow = alow[:pad_start_idx]

            # if clean_special_tokens:
            #     # remove <<stop>> tokens
            #     if self.stop_token in alow:
            #         stop_start_idx = alow.index(self.stop_token)
            #         alow = alow[:stop_start_idx]

            word_action = None
            num_action = None
            if alow[0][0] == 0:
                word_action = 'stop'
                num_action = [0]
            elif alow[0][0] == 1: #base
                if alow[0][1] == 0:
                    word_action = 'NoOp' #made by me
                if alow[0][1] == 1:
                    word_action = 'MoveAhead'
                elif alow[0][1] == 2:
                    word_action = 'MoveBack'
                elif alow[0][1] == 3:
                    word_action = 'MoveRight'
                elif alow[0][1] == 4:
                    word_action = 'MoveLeft'
                num_action = alow[0][1]
            elif alow[0][0] == 2: #rotate
                word_action = "RotateAgent"
                num_action = alow[0][2]
            elif alow[0][0] == 3: #arm
                word_action = 'MoveArm'
                num_action = alow[0][3:6]
            elif alow[0][0] == 4: #ee
                if alow[0][6] == 0:
                    word_action = 'NoOp' #made by me
                elif alow[0][6] == 1:
                    word_action = 'PickupObject'
                elif alow[0][6] == 2:
                    word_action = 'ReleaseObject'
                num_action = alow[0][6]
            elif alow[0][0] == 5: #look
                if alow[0][7] == 0:
                    word_action = 'NoOp'
                if alow[0][7] == 1:
                    word_action = 'LookUp'
                elif alow[0][7] == 2:
                    word_action = 'LookDown'
                num_action = alow[0][7]

            task_id_ann = ex['root'].split('/')[-1]
            pred[task_id_ann] = {
                'action_low_word': word_action,
                'action_low_num': num_action,
                'action_low': alow
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action for real-time inference
        '''
        # device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        # action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        # action_emb = self.dec.emb(action_num).unsqueeze(0)
        
        # embedding to input into next step in the sequence
        embedded_mode = self.dec.emb['emb_mode'](action[:, :1])
        embedded_base = self.dec.emb['emb_base'](action[:, 1:2])  
        embedded_yaw = self.dec.emb['emb_yaw'](action[:, 2:3])  
        embedded_eff_xyz = self.dec.emb['emb_eff_xyz'](action[:, 3:6])  
        embedded_grasp_drop = self.dec.emb['emb_grasp_drop'](action[:, 6:7])
        embedded_up_down = self.dec.emb['emb_up_down'](action[:, 7:8])  
        embedded_actions = torch.cat([embedded_mode, embedded_base, embedded_yaw, embedded_eff_xyz, embedded_grasp_drop, embedded_up_down], dim=1)
        action_emb = embedded_actions.view(embedded_actions.size(0), -1) #flatten

        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        losses = dict()

        # GT and predictions
        if self.args.class_mode:
            action_logits = out['out_action_low']
            #get top predicted action from distribution
            logits_1d = action_logits[:, :, :(1*self.mode_class_num)].view(-1, 1, self.mode_class_num)
            last = (1*self.mode_class_num)
            logits_base_1d = action_logits[:, :, last: last+(1*self.base_class_num)].view(-1, 1, self.base_class_num)
            last = last + (1*self.base_class_num)
            logits_yaw_1d = action_logits[:, :, last: last+(1*self.dec.num_bins)].view(-1, 1, self.dec.num_bins)
            last = last + (1*self.dec.num_bins)
            logits_3d = action_logits[:, :, last : last+(3 * self.dec.num_bins)].view(-1, 3, self.dec.num_bins)
            last = last + (3 * self.dec.num_bins)
            logits_2d = action_logits[:, :, last : last + (2 * self.grasp_drop_class_num)].view(-1, 2, self.grasp_drop_class_num)
        else:
            p_alow = out['out_action_low'].view(-1, self.args.action_dims)
        
        # ignore the warning in the line below since it's just torch scalars being converted to tensors
        l_alow = []  # Initialize an empty list to store the concatenated tensors

        # Iterate over each sublist in feat['action_low']
        l_alow = []
        for sublist in feat['action_low']:
            for i, item in enumerate(sublist):
                temp_tensor = torch.cat([
                    item['mode'].unsqueeze(0),
                    item['base_action'].unsqueeze(0),
                    item['state_rot'] if item['state_rot'].dim() == 1 else item['state_rot'].unsqueeze(0),
                    # item['state_rot'].unsqueeze(0),
                    item['state_ee'] if item['state_ee'].dim() == 1 else item['state_ee'].unsqueeze(0),
                    item['grasp_drop'].unsqueeze(0),
                    item['up_down'].unsqueeze(0)
                ], dim=0).to(device)
                l_alow.append(temp_tensor)
        l_alow = torch.stack(l_alow)
    
        # action loss
        pad_tensor = torch.full_like(l_alow, self.action_pad) # -1 is the action pad index for class mode
        pad_valid = (l_alow != pad_tensor).all(dim=1) #collapse the bools in the inner tensors to 1 bool
        p_alow_1d_valid = logits_1d[pad_valid]
        p_alow_1d_base_valid = logits_base_1d[pad_valid]
        p_alow_1d_yaw_valid = logits_yaw_1d[pad_valid]
        p_alow_3d_valid = logits_3d[pad_valid]
        p_alow_2d_valid = logits_2d[pad_valid]
        l_alow_valid = l_alow[pad_valid]

        if self.args.class_mode:
            total_loss = torch.zeros(l_alow_valid.shape[0]).to('cuda')
            for dim in range(l_alow_valid.shape[1]): #loops 8 times, one for each action dim
                if dim == 0:
                    loss = nn.CrossEntropyLoss(reduction='none')(p_alow_1d_valid[:, dim, :], l_alow_valid[:, dim])
                elif dim == 1:
                    loss = nn.CrossEntropyLoss(reduction='none')(p_alow_1d_base_valid[:, dim-1, :], l_alow_valid[:, dim])
                elif dim == 2:
                    loss = nn.CrossEntropyLoss(reduction='none')(p_alow_1d_yaw_valid[:, dim-2, :], l_alow_valid[:, dim])
                elif dim > 2 and dim < 6:
                    loss = nn.CrossEntropyLoss(reduction='none')(p_alow_3d_valid[:, dim-3, :], l_alow_valid[:, dim])
                else:
                    loss = nn.CrossEntropyLoss(reduction='none')(p_alow_2d_valid[:, dim-6, :], l_alow_valid[:, dim])
                total_loss += loss #add all action dims losses together for each step
            alow_loss = total_loss / l_alow_valid.shape[1] #avg loss for all action dims losses for each trajectory (each step's total across all dims is divided by num of dims)
        else:
            alow_loss = F.mse_loss(p_alow, l_alow, reduction='none') #regression
        # Calculate the mean loss only over valid elements
        alow_loss_mean = alow_loss.mean()
        losses['action_low'] = alow_loss_mean * self.args.action_loss_wt
        return losses

    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
        return {k: sum(v)/len(v) for k, v in m.items()}