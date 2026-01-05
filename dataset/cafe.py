"""
https://github.com/dk-kim/CAFE_codebase
"""
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import json
import numpy as np
import random
from PIL import Image

Activity_names = ['Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie']

TRAIN_CAFE_P = ['1', '2', '3', '4', '9', '10', '11', '12', '17', '18', '19', '20', '21', '22', '23', '24']
VAL_CAFE_P = ['13', '14', '15', '16']
TEST_CAFE_P = ['5', '6', '7', '8']

TRAIN_CAFE_V = ['1', '2', '5', '6', '9', '10', '13', '14', '17', '18', '21', '22']
VAL_CAFE_V = ['3', '7', '11', '15', '19', '23']
TEST_CAFE_V = ['4', '8', '12', '16', '20', '24']

def cafe_path(img_root, ann_root):


    train_img_path = [img_root / f'image_{str(i*2)}' / train_seq for i in range(5) for train_seq in train_seqs]
    val_img_path = [img_root / f'image_{str(i*2)}' / val_seq for i in range(5) for val_seq in val_seqs]

    train_ann_path = [ann_root / f'{train_seq}_image{str(i*2)}.json' for train_seq in train_seqs for i in range(5)]
    val_ann_path = [ann_root / f'{val_seq}_image{str(i*2)}.json' for val_seq in val_seqs for i in range(5)]


    PATHS = {
        "train": train_ann_path,
        "val": val_ann_path
    }

    train_ann_file = PATHS['train']
    test_ann_file = PATHS['val']  # imgs and anns paths

    return train_ann_file, test_ann_file