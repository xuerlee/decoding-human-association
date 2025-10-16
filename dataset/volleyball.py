import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import random
import sys
from pathlib import Path

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9

def volleyball_path(img_root, ann_root):
    # train_seqs = [str(i + 1) for i in range(32)]
    # val_seqs = [str(i + 1) for i in range(32)]
    # val_seqs = [str(i + 33) for i in range(12)]
    # val_seqs = [str(44)]

    # for testing code runing one seq
    # train_seqs = [str(i + 1) for i in range(1)]
    # val_seqs = [str(i + 1) for i in range(1)]

    # random seqs
    # all_seqs = [str(i + 1) for i in range(44)]
    # random.shuffle(all_seqs)
    # train_seqs = all_seqs[:32]
    # val_seqs = all_seqs[32:]

    # DIN seqs
    val_seqs = [5, 6, 7, 8, 9, 10, 11, 15, 16, 25, 28, 29]
    val_seqs = [str(s) for s in val_seqs]
    train_seqs = [str(s) for s in range(1, 45) if s not in val_seqs]
    # val_seqs = train_seqs

    train_seq_path = [img_root / ('seq' + train_seq.zfill(2)) for train_seq in train_seqs]
    val_seq_path = [img_root / ('seq' + val_seq.zfill(2)) for val_seq in val_seqs]
    train_ann_path = [ann_root / (train_seq + '_annotations.txt') for train_seq in train_seqs]
    val_ann_path = [ann_root / (val_seq + '_annotations.txt') for val_seq in val_seqs]

    train_img_path = [file for seq in train_seq_path for file in seq.rglob("*")]
    val_img_path = [file for seq in val_seq_path for file in seq.rglob("*")]  # find all folders and files inside

    PATHS = {
        "train": (train_img_path, train_ann_path),
        "val": (val_img_path, val_ann_path),
    }

    train_img_file, train_ann_file = PATHS['train']
    test_img_file, test_ann_file = PATHS['val']  # imgs and anns paths

    return train_img_file, train_ann_file, test_img_file, test_ann_file