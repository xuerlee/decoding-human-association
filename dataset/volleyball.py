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

'''
Reference:
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark
'''

'''
Action_names: ['Waiting', 'Setting', 'Digging', 'Falling', 'Spiking', 'Blocking', 'Jumping', 'Moving', 'Standing']
Activity_names: ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass', 'l_winpoint']
[Pass]: Players who are trying an underhand pass independently of whether or not they are successful.
[Set]: Player who is doing an overhand pass and those who are going to spike the ball whether they are really trying or faking.
[Spike]: Players who are spiking and blocking. 
[Winpoint]: All players in the team that scores a point. This group activity is observed for a few seconds right after the score.
'''


def volleyball_path(img_root, ann_root):
    # DIN seqs
    train_seqs = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54,
                       0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    val_seqs = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
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