import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
import clip
import json
import random

ORIGINAL_STATIC_RES = 200
ORIGINAL_GRIPPER_RES = 84

class DataPrefetcher():
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # Dataloader will prefetch data to cpu so this step is very quick
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        with torch.cuda.stream(self.stream):
            for key in self.batch:
                self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if batch[key] is not None:
                    batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time

class LMDBDataset(Dataset):
    def __init__(self, lmdb_dir, sequence_length, chunk_size, action_mode, action_dim, start_ratio, end_ratio, unseen_train=False,history_length=None):
        super(LMDBDataset).__init__()
        self.sequence_length = sequence_length
        if history_length is None:
            self.history_length = sequence_length
        else:
            self.history_length = history_length
        self.history_length = history_length
        self.chunk_size = chunk_size
        self.action_mode = action_mode
        self.action_dim = action_dim
        self.dummy_rgb_static = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES, ORIGINAL_STATIC_RES, dtype=torch.uint8)
        self.dummy_rgb_gripper = torch.zeros(sequence_length, 3, ORIGINAL_GRIPPER_RES, ORIGINAL_GRIPPER_RES, dtype=torch.uint8)
        self.dummy_arm_state = torch.zeros(sequence_length, 6)
        self.dummy_gripper_state =  torch.zeros(sequence_length, 2)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, action_dim)
        self.dummy_mask = torch.zeros(sequence_length, chunk_size)
        self.dummy_joint_state = torch.zeros(sequence_length, 7)
        self.lmdb_dir = lmdb_dir
        env = lmdb.open(lmdb_dir, readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio) 
            self.end_step = int(dataset_len * end_ratio) - sequence_length - chunk_size
        env.close()
        self.unseen_train = unseen_train
        if unseen_train:
            with open('data/task_unseen/unseen_train.json', 'r') as f:
                self.unseen_inst_str = json.load(f)

    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_dir, readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()
        idx = idx + self.start_step
        
        rgb_static = self.dummy_rgb_static.clone() # chunk_size, 3, 200, 200
        rgb_gripper = self.dummy_rgb_gripper.clone() # chunk_size, 3, 84, 84
        arm_state = self.dummy_arm_state.clone() # chunk_size, 6
        gripper_state = self.dummy_gripper_state.clone() # chunk_size, 2
        actions = self.dummy_actions.clone()
        mask = self.dummy_mask.clone()
        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))
        try:
            inst_token = loads(self.txn.get(f'inst_token_{cur_episode}'.encode()))
        except:
            inst_token = None
        if inst_token is None or self.unseen_train:
            # clip tokenize
            inst_str = loads(self.txn.get(f'inst_{cur_episode}'.encode()))
            if self.unseen_train:
                # use rewrite the instruction
                cand_inst_str_list = self.unseen_inst_str[inst_str]
                inst_str = random.choice(cand_inst_str_list)
            inst_token = clip.tokenize(inst_str)[0]
        for i in range(self.sequence_length):
            if loads(self.txn.get(f'cur_episode_{idx+i}'.encode())) == cur_episode:
                rgb_static[i] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx+i}'.encode())))
                rgb_gripper[i] = decode_jpeg(loads(self.txn.get(f'rgb_gripper_{idx+i}'.encode())))
                robot_obs = loads(self.txn.get(f'robot_obs_{idx+i}'.encode()))
                arm_state[i, :6] = robot_obs[:6]
                gripper_state[i, ((robot_obs[-1] + 1) / 2).long()] = 1
                for j in range(self.chunk_size):
                    if loads(self.txn.get(f'cur_episode_{idx+i+j}'.encode())) == cur_episode:
                        mask[i, j] = 1
                        if self.action_mode == 'ee_rel_pose':
                            actions[i, j] = loads(self.txn.get(f'rel_action_{idx+i+j}'.encode()))
                        elif self.action_mode == 'ee_abs_pose':
                            actions[i, j] = loads(self.txn.get(f'abs_action_{idx+i+j}'.encode()))
                        actions[i, j, -1] = (actions[i, j, -1] + 1) / 2
        return {
            'rgb_static': rgb_static,
            'rgb_gripper': rgb_gripper,
            'inst_token': inst_token,
            'arm_state': arm_state,
            'gripper_state': gripper_state,
            'actions': actions,
            'mask': mask,
        }

    def __len__(self):
        return self.end_step - self.start_step
