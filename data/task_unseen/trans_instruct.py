# Modified from https://github.com/EDiRobotics/GR1-Training/blob/b0c0fdb52787521ee7c5856481154f58318a37bd/calvin2lmdb.py
import os
import io
import argparse
from pickle import dumps, loads
import numpy as np
import torch
from torchvision.io import encode_jpeg
from einops import rearrange
from tqdm import tqdm


def get_instruct(input_dir):
    instruct_set = set()
    annotations = np.load(os.path.join(input_dir, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True).tolist()['language']['ann']
    start_end_ids = np.load(os.path.join(input_dir, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True).tolist()['info']['indx']

    for index, (start, end) in tqdm(enumerate(start_end_ids), total=len(start_end_ids), desc='Processing episodes'):
        inst = annotations[index]
        instruct_set.add(inst)
    return list(instruct_set)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer CALVIN dataset to lmdb format.")
    parser.add_argument("--input_dir", default='/data/apps/task_ABCD_D', type=str, help="Original dataset directory.")
    parser.add_argument("--output_file", default='/data/apps/AR_VLA/data/task_unseen/unseen_train.json', type=str, help="Original dataset directory.")
    args = parser.parse_args()
    print(f'Input directory: {args.input_dir}')
    if args.output_dir is None:
        base_dir = os.path.basename(args.input_dir)
        args.output_dir = os.path.join('data', base_dir)
    print(f'Output file: {args.output_file}')
    train_instruct_list = get_instruct( os.path.join(args.input_dir, 'training'))
    