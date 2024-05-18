import torch
from torch import nn
from torch.nn import functional as F


class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64*7*7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


class ConvFrameMaskDecoder(nn.Module):
    '''
    action decoder
    '''

    def __init__(self, max_vals, min_vals, emb, num_bins, class_mode, demb, dframe, dhid, action_dims, mode_class_num, grasp_drop_class_num, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0., adapter_dropout=0,
                 teacher_forcing=False):
        super().__init__()

        self.emb = emb
        self.max_vals = max_vals
        self.min_vals = min_vals
        self.class_mode = class_mode
        self.num_bins = num_bins + 2
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.adapter_dropout = nn.Dropout(adapter_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.action_dims = action_dims
        self.mode_class_num = mode_class_num
        self.grasp_drop_class_num = grasp_drop_class_num
        if self.class_mode: #action layer for classification
            #FIXME change dim nums
            self.actor = nn.Linear(dhid + dhid + dframe + demb, (1*self.mode_class_num)+((self.action_dims-3) * self.num_bins) + (2*self.grasp_drop_class_num))
            # self.actor2 = nn.Linear(dhid + dhid + dframe + demb,, 2 * 5) # 2 dimensions, 5 classes each
        else: #action layer for regression
            self.actor = nn.Linear(dhid + dhid + dframe + demb, self.action_dims)
        if self.class_mode:
            self.adapter = nn.Linear(self.action_dims * demb, demb) #reduce dim size
        else: # only for regression since classification has embeddings of proper size
            self.adapter = nn.Linear(self.action_dims, demb) #enlarge dim size
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        self.flag = False

        # nn.init.uniform_(self.go, -0.1, 0.1)
        nn.init.xavier_uniform_(self.adapter.weight) #maybe initialize with custom range later but probably not needed
        
        if self.class_mode: #classification
            #initialize each dimension's weights to a random value within its actual value range
            # prev = 0
            # num_bins_iter = self.num_bins
            # for dim in range(action_dim):
            #     nn.init.uniform_(self.actor.weight[prev:num_bins_iter, :], a=self.min_vals[dim], b=self.max_vals[dim])
            #     prev += self.num_bins
            #     num_bins_iter += self.num_bins 
            
            #initialize weights to random values within the ranges
            #FIXME change dim nums
            nn.init.uniform_(self.actor.weight[:(1*self.mode_class_num), :], a=0, b=self.mode_class_num) # b is exlusive
            last = (1*self.mode_class_num)
            nn.init.uniform_(self.actor.weight[last:last+((self.action_dims-3) * self.num_bins) , :], a=0, b=self.num_bins)
            last = last+((self.action_dims-3) * self.num_bins)
            nn.init.uniform_(self.actor.weight[last:last+((self.action_dims-7) * self.grasp_drop_class_num), :], a=0, b=self.grasp_drop_class_num)
        else: #regression
            nn.init.xavier_uniform_(self.actor.weight) #initialize with custom range later


    def step(self, enc, frame, e_t, state_tm1):
        # previous decoder hidden state
        h_tm1 = state_tm1[0] #tm1 = t minus 1

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))
        
        #skip the first LSTM cell, reduce dim for 2nd LSTM cell and beyond
        if self.flag:
            e_t = self.adapter(self.adapter_dropout(e_t))
        self.flag = True

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1) #input for LSTM
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1) # pass concatenated input along with the previous hidden state from LSTM into LSTM. returns next hidden state
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t = state_t[0] #updates the hidden state

        # decode action
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        #action_t = action_emb_t.mm(self.emb.weight.t()) #decode the action distribution for each traj in a batch (old discrete)
        if self.class_mode: #classification
            # logits = action_emb_t.view(-1, self.action_dim, self.num_bins)
            # action_t = logits
            action_t = action_emb_t
        else: #regression
            action_t = action_emb_t
        return action_t, state_t, lang_attn_t

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None): #max_decode = the max num of actions to predict
        max_t = len(gold[0]) #if self.training else min(max_decode, frames.shape[1]) # the num of actions to predict
        batch = enc.size(0) #batch size
        e_t = self.go.repeat(batch, 1) #batch num of SOS action embeddings
        state_t = state_0

        actions = [] # all predicted action distributions for every step for every trajectory in the batch
        attn_scores = []
        for t in range(max_t):
            action_t, state_t, attn_score_t = self.step(enc, frames[:, t], e_t, state_t) #does 1 forward pass thorugh the network for all traj in the batch. returns batch num action distributions
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                if self.class_mode:
                    #FIXME change dim nums
                    logits_1d = action_t[:, :(1*self.mode_class_num)].view(-1, 1, self.mode_class_num)
                    last = (1*self.mode_class_num)
                    logits_6d = action_t[:, last : last+((self.action_dims-3) * self.num_bins)].view(-1, 6, self.num_bins)
                    last = last +((self.action_dims-3) * self.num_bins)
                    logits_2d = action_t[:, last :  last + (2 * self.grasp_drop_class_num)].view(-1, 2, self.grasp_drop_class_num)
                    w_t_1d = logits_1d.max(dim=2)[1]
                    w_t_6d = logits_6d.max(dim=2)[1]
                    w_t_2d = logits_2d.max(dim=2)[1]
                    w_t = torch.cat((w_t_1d, w_t_6d, w_t_2d), dim=1)
                else:
                    w_t = action_t  # No need to find max index, assume action_t gives continuous output directly

            if self.class_mode: #classification
                # embedding to input into next step in the sequence
                #FIXME change dim nums
                embedded_mode = self.emb['emb_mode'](w_t[:, :1])
                embedded_xy = self.emb['emb_xy'](w_t[:, 1:3])  
                embedded_yaw = self.emb['emb_yaw'](w_t[:, 3:4])  
                embedded_eff_xyz = self.emb['emb_eff_xyz'](w_t[:, 4:7])  
                embedded_grasp_drop = self.emb['emb_grasp_drop'](w_t[:, 7:8])
                embedded_up_down = self.emb['emb_up_down'](w_t[:, 8:9])  
                embedded_actions = torch.cat([embedded_mode, embedded_xy, embedded_yaw, embedded_eff_xyz, embedded_grasp_drop, embedded_up_down], dim=1)
                e_t = embedded_actions.view(embedded_actions.size(0), -1) #flatten
            else: #regression
                e_t = w_t
        
        results = {
            # 'out_action_low': torch.stack(actions, dim=1).flatten(start_dim=2) if self.class_mode else torch.stack(actions, dim=1), #reshapes it group each trajectory's steps' distributions together. new shape is [batch, steps, action_space] (groups steps within the same traj together)
            'out_action_low': torch.stack(actions, dim=1), #reshapes it group each trajectory's steps' distributions together. new shape is [batch, steps, action_space] (groups steps within the same traj together)
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t
        }
        self.flag = False
        return results