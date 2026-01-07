"""
Reference:
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark
"""
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


Action_names = ['blocking', 'digging', 'falling', 'jumping',
                'moving', 'setting', 'spiking', 'standing',
                'waiting']
Activity_names = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']
'''
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

    train_ann_path = [img_root / str(train_seq) / 'annotations.txt' for train_seq in train_seqs]
    val_ann_path = [img_root / str(val_seq) / 'annotations.txt' for val_seq in val_seqs]

    PATHS = {
        "train": train_ann_path,
        "val": val_ann_path
    }

    train_ann_file = PATHS['train']
    test_ann_file = PATHS['val']  # imgs and anns paths

    return train_ann_file, test_ann_file


def volleyball_read_annotations(ann_file, ann_social_folder):
    annotations = {}  # annotations for each frame

    gact_to_id = {name: i for i, name in enumerate(Activity_names)}
    act_to_id = {name: i for i, name in enumerate(Action_names)}

    with open(ann_file, 'r') as ann_txt:
        se_anns = ann_txt.readlines()
        for se_ann in se_anns:
            values = se_ann[:-1].split(' ')
            frame_id = int(values[0].split('.jpg')[0])

            annotations[frame_id] = {}
            annotations[frame_id]['groups'] = []
            annotations[frame_id]['persons'] = []

            group_id = 0
            activity = gact_to_id[values[1]]
            annotations[frame_id]['groups'].append({
                'group_id': group_id,
                'activity': activity,
                'include_id': []
            })

            ann_social_path = ann_social_folder + f'/{str(frame_id)}/{str(frame_id)}.txt'
            with open(ann_social_path, 'r') as ann_social_txt:
                social_anns = ann_social_txt.readlines()
                for social_ann in social_anns:
                    values_social = social_ann[:-2].split(' ')
                    frame_id_social = int(values_social[5])
                    if frame_id == frame_id_social:
                        person_id = int(values_social[0])
                        x1 = float(values_social[1])
                        y1 = float(values_social[2])
                        x2 = float(values_social[3])
                        y2 = float(values_social[4])  # absolute coord
                        bbox = [x1, y1, x2, y2]
                        action = act_to_id[values_social[-1]]
                        if_ingroup = int(values_social[7])
                        if if_ingroup == 1:
                            annotations[frame_id]['persons'].append({
                                'person_id': person_id,
                                'bbox': bbox,
                                'action': action,
                                'group_id': 0
                            })
                            annotations[frame_id]['groups'][0]['include_id'].append(person_id)
                        else:
                            annotations[frame_id]['persons'].append({
                                'person_id': person_id,
                                'bbox': bbox,
                                'action': action,
                                'group_id': -1
                            })

            annotations[frame_id]['persons'].sort(key=lambda x: x['person_id'])
    # print(annotations)
    return annotations


def volleyball_read_dataset(ann_files, args):
    data = {}
    for ann_file in ann_files:
        sid = int(str(ann_file).split('/')[-2])
        ann_social_folder = args.ann_path + f'/{str(sid)}'
        data[sid] = volleyball_read_annotations(ann_file, ann_social_folder)  # data for each seq
    return data


'''
data structure:
data: {1: {ann1}, 2: {ann2}, ... , seq: {annseq}}
ann1(seq): {1: {ann1-1}, 2: {ann1-2}, ... , frame: {ann1-frame}}
ann1(seq)-1(frame): {persons: [{person_id: 1, bbox: [], action:, 1, group_id: 1}, ..., {...}], 
        groups: [{group_id: 1, activity: 1, person_ids: []}, ..., {...}]}
'''


def volleyball_all_frames(anns, num_frames):
    half_left = num_frames // 2
    half_right = num_frames - half_left
    return [(s, f) for s in anns for f in anns[s]]
    # (sid, fid) with anns (every 10 frames: eg. 11, 21, 31, ...)
    # filtered the first and the last anns


class Volleyball_Dataset(data.Dataset):
    def __init__(self, anns, frames, img_path, transform, num_frames=10, is_training=True):
        """
        Args:
            Characterize collective dataset based on feature maps.
        """
        self.anns = anns
        self.frames = frames
        self.img_path = img_path

        self.num_frames = num_frames  # number of stacked frame features
        self.is_training = is_training

        self.transform = transform

    def __len__(self):
        return len(self.frames)  # number of frames with anns
        # TODO: put key frame in middle and take 10 frames currently
        #  put key frames at the third and seventh location later

    def __getitem__(self, idx):
        # Load feature map

        select_frames = self.get_frames(self.frames[idx])  # num_frames
        sample = self.load_samples_sequence(select_frames)

        return sample

    def get_frames(self, frame):
        sid, src_fid = frame

        if self.is_training:
            half_left = self.num_frames // 2
            half_right = self.num_frames - half_left

            fids = range(src_fid - half_left,
                         src_fid + half_right)

            return sid, src_fid, [(sid, src_fid, fid) for fid in fids]
            # normal training: each training loading 10 frames
        else:
            half_left = self.num_frames // 2
            half_right = self.num_frames - half_left

            fids = range(src_fid - half_left,
                         src_fid + half_right)

            return sid, src_fid, [(sid, src_fid, fid) for fid in fids]            # normal testing: each test loading 10 frames

    def load_samples_sequence(self, select_frames):
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda')
    #     else:
    #         device = torch.device('cpu')

        sid = select_frames[0]
        src_fid = select_frames[1]

        person_ids = []
        bboxes = []
        actions = []
        # p_group_ids = []

        group_ids = []
        activities = []
        include_ids = []

        for person in self.anns[sid][src_fid]['persons']:
            person_id = person['person_id']  # to connect the prediction and build pred matrix for iou group-level loss of group members
            person_ids.append(person_id)  # it is not in order in the tensor, following the order of group id, and some disappear
            bbox = person['bbox']  # for ROI Align and positional encoding
            bboxes.append(bbox)
            action = person['action']  # gt individual actions
            actions.append(action)
            # p_group_id = person['group_id']  # gt for person-level cross-entropy loss of group members
            # p_group_ids.append(p_group_id)

        for group in self.anns[sid][src_fid]['groups']:
            activity = group['activity']  # gt for group activity
            activities.append(activity)
            # group_id = group['group_id']
            # group_ids.append(group_id)
            include_id = group['include_id']
            include_ids.append(include_id)

        num_persons = len(person_ids)
        num_groups = len(activities)
        print(num_persons)

        one_hot_matrix = np.zeros((num_persons, num_groups), dtype=float)
        # TODO: check if the order of column and the row should change
        person_to_index = {p: i for i, p in enumerate(person_ids)}

        for group, persons in enumerate(include_ids):
            print(include_ids)
            for person in persons:
                one_hot_matrix[person_to_index[person], group] = 1

        imgs = []
        bbox = bboxes.copy()
        bbox = np.array(bbox, dtype=np.float64).reshape(-1, 4)
        for i, (sid, src_fid, fid) in enumerate(select_frames[2]):  # 10 frames for 1 item
            img = cv2.imread(self.img_path + f'/{sid}/{src_fid}/{fid}.jpg')[:, :, [2, 1, 0]]  # BGR -> RGB  # H, W, 3
            img = Image.fromarray(img)
            img, new_bboxes = self.transform(img, bbox)  # bboxes did not changed here
            imgs.append(img)

        # labels for the whole video clip (the label of the key frame)
        meta = {}
        imgs = np.stack(imgs)
        # bboxes = np.array(bboxes, dtype=np.float64).reshape(-1, 4)
        actions = np.array(actions, dtype=np.int32)
        activities = np.array(activities, dtype=np.int32)
        one_hot_matrix = np.array(one_hot_matrix, dtype=np.int32)

        imgs = torch.from_numpy(imgs).float()
        imgs = torch.squeeze(imgs, 1)  # shape: (10, 3, H, W)
        bboxes = torch.from_numpy(new_bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        one_hot_matrix = torch.from_numpy(one_hot_matrix).int()

        meta['sid'] = sid
        meta['src_fid'] = src_fid
        meta['frame_size'] = (720, 1280)

        return imgs, bboxes, actions, activities, one_hot_matrix, meta
