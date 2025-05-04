import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

import matplotlib.pyplot as plt
import random
import sys
from pathlib import Path

FRAMES_SIZE = {1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720),
               6: (480, 720), 7: (480, 720), 8:  (480, 720), 9:  (480, 720), 10: (480, 720),
               11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800),
               16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800),
               21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720),
               26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720),
               31: (480, 720),  32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720),
               36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720),
               41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}

def collective_path(fm_root, ann_root):
    train_seqs = [str(i + 1) for i in range(32)]
    val_seqs = [str(i + 33) for i in range(12)]
    train_seq_path = [fm_root / ('seq' + train_seq.zfill(2)) for train_seq in train_seqs]
    val_seq_path = [fm_root / ('seq' + val_seq.zfill(2)) for val_seq in val_seqs]
    train_ann_path = [ann_root / (train_seq + '_annotations.txt') for train_seq in train_seqs]
    val_ann_path = [ann_root / (val_seq + '_annotations.txt') for val_seq in val_seqs]

    train_fm_path = [file for seq in train_seq_path for file in seq.rglob("*")]
    val_fm_path = [file for seq in val_seq_path for file in seq.rglob("*")]  # find all folders and files inside

    PATHS = {
        "train": (train_fm_path, train_ann_path),
        "val": (val_fm_path, val_ann_path),
    }

    train_fm_file, train_ann_file = PATHS['train']
    test_fm_file, test_ann_file = PATHS['val']  # feature maps and anns paths

    # print(train_fm_file, train_ann_file, test_fm_file, test_ann_file)

    return train_fm_file, train_ann_file, test_fm_file, test_ann_file


def collective_read_annotations(ann_file):
    annotations = {}  # annotations for each frame
    with open(ann_file, 'r') as ann_txt:
        se_anns = ann_txt.readlines()
        for se_ann in se_anns:
            se_ann = se_ann.rstrip()
            frame_id = int(se_ann.split('\t')[0])
            if frame_id not in annotations:
                annotations[frame_id] = {}
                annotations[frame_id]['groups'] = []
                annotations[frame_id]['persons'] = []
            x1 = float(se_ann.split('\t')[1])
            y1 = float(se_ann.split('\t')[2])
            x2 = float(se_ann.split('\t')[3])
            y2 = float(se_ann.split('\t')[4])  # absolute coord
            bbox = [x1, y1, x2, y2]
            action = int(se_ann.split('\t')[5]) - 1  # start with 0
            person_id = int(se_ann.split('\t')[7]) - 1
            group_id = int(se_ann.split('\t')[8]) - 1

            if any(person.get('group_id') == group_id for person in annotations[frame_id]['persons']):
                group = [group for group in annotations[frame_id]['groups'] if group.get('group_id') == group_id][0]
                group['include_id'].append(person_id)
            else:
                activity = int(se_ann.split('\t')[6]) - 1
                annotations[frame_id]['groups'].append({
                                'group_id': group_id,
                                'activity': activity,
                                'include_id': [person_id]
                            })

            annotations[frame_id]['persons'].append({
                            'person_id': person_id,
                            'bbox': bbox,
                            'action': action,
                            'group_id': group_id
                        })
    # print(annotations)
    return annotations


def collective_read_dataset(ann_files):
    data = {}
    for ann_file in ann_files:
        sid = int(str(ann_file).split('/')[-1].split('_')[0])
        data[sid] = collective_read_annotations(ann_file)  # data for each seq
    return data


'''
data stucture:
data: {1: {ann1}, 2: {ann2}, ... , seq: {annseq}}
ann1(seq): {1: {ann1-1}, 2: {ann1-2}, ... , frame: {ann1-frame}}
ann1(seq)-1(frame): {persons: [{person_id: 1, bbox: [], action:, 1, group_id: 1}, ..., {...}], 
        groups: [{group_id: 1, activity: 1, person_ids: []}, ..., {...}]}
'''


def collective_all_frames(anns):
    return [(s, f) for s in anns for f in anns[s] if f != 1 and f != max(anns[s])]
    # (sid, fid) with anns (every 10 frames: eg. 11, 21, 31, ...)
    # filtered the first and the last anns


class FeatureMapDataset(data.Dataset):
    def __init__(self, anns, frames, feature_path, feature_size, num_frames=10, is_training=True):
        """
        Args:
            Characterize collective dataset based on feature maps.
        """
        self.anns = anns
        self.frames = frames
        self.feature_path = feature_path
        self.feature_size = feature_size  # feature size output by the feature extraction part

        self.num_frames = num_frames  # number of stacked frame features
        self.is_training = is_training

        # TODO: match feature paths with ann, match person ids, roi align

    def __len__(self):
        return len(self.frames)
        # TODO: put key frame in middle and take 10 frames currently
        #  put key frames at the third and seventh location later

    def __getitem__(self, idx):
        # Load feature map

        select_frames = self.get_frames(self.frames[idx])  # 10 frames
        sample = self.load_samples_sequence(select_frames)

        return sample

    def get_frames(self, frame):
        sid, src_fid = frame

        if self.is_training:
            return [(sid, src_fid, fid) for fid in range(src_fid - int(self.num_frames / 2), src_fid + int(self.num_frames / 2))]
            # normal training: each training loading 10 frames
        else:
            return [(sid, src_fid, fid) for fid in range(src_fid - int(self.num_frames / 2), src_fid + int(self.num_frames / 2))]
            # normal testing: each test loading 10 frames

    def load_samples_sequence(self, select_frames):
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda')
    #     else:
    #         device = torch.device('cpu')
        featuremaps = []
        person_ids = []
        bboxes = []
        actions = []
        p_group_ids = []

        group_ids = []
        activities = []
        include_ids = []

        for i, (sid, src_fid, fid) in enumerate(select_frames):  # 10 frames for 1 item
            print('selected frames:', sid, src_fid, fid)
            feature_map = torch.from_numpy(torch.load(self.feature_path + '/seq%02d/frame%04d_features.pt' % (sid, fid))).float()
            # feature_map = feature_map.to(device=device)  # can't do:  Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
            # TODO: (errors now) check if it can be put on CUDA BEFORE dataloader or maybe collate_fn
            featuremaps.append(feature_map)

            for person in self.anns[sid][src_fid]['persons']:
                person_id = person['person_id']  # to connect the prediction and build pred matrix for iou group-level loss of group members
                print('person_id:', person_id)
                person_ids.append(person_id)  # it is not in order in the tensor, following the order of group id, and some disappear
                bbox = person['bbox']  # only for ROI Align
                bboxes.append(bbox)
                action = person['action']  # gt individual actions
                actions.append(action)
                p_group_id = person['group_id']  # gt for person-level cross-entropy loss of group members
                print('p_group_id:', p_group_id)
                p_group_ids.append(p_group_id)

            for group in self.anns[sid][src_fid]['groups']:
                group_id = group['group_id']
                print('group_id:', group_id)
                group_ids.append(group_id)
                activity = group['activity']  # gt for group activity
                activities.append(activity)
                include_id = group['include_id']
                include_ids.append(include_id)

        featuremaps = torch.cat(featuremaps)
        # print(featuremaps.shape)  # shape: (10, 1392, 31, 46)
        # OH = featuremaps.shape[2]
        # OW = featuremaps.shape[3]

        bboxes = torch.cat(bboxes)

        return featuremaps, bboxes

    def collate_fn(self, batch):
        # TODO: padding_mask, roi align, CUDA
        pass