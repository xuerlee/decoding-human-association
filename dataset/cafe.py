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
import cv2
from PIL import Image

Action_names = ['Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie', 'Individual']
Activity_names = ['Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie', 'Individual']

TRAIN_CAFE_P = ['1', '2', '3', '4', '9', '10', '11', '12', '17', '18', '19', '20', '21', '22', '23', '24']
VAL_CAFE_P = ['13', '14', '15', '16']
TEST_CAFE_P = ['5', '6', '7', '8']

TRAIN_CAFE_V = ['1', '2', '5', '6', '9', '10', '13', '14', '17', '18', '21', '22']
VAL_CAFE_V = ['3', '7', '11', '15', '19', '23']
TEST_CAFE_V = ['4', '8', '12', '16', '20', '24']

def cafe_path(img_root, ann_root, split):

    if split == 'place':
        train_path = [img_root / str(train_seq) for train_seq in TRAIN_CAFE_P]
        val_path = [img_root / str(val_seq) for val_seq in VAL_CAFE_P]
        test_path = [img_root / str(test_seq) for test_seq in TEST_CAFE_P]

    elif split == 'view':
        train_path = [img_root / str(train_seq) for train_seq in TRAIN_CAFE_V]
        val_path = [img_root / str(val_seq) for val_seq in VAL_CAFE_V]
        test_path = [img_root / str(test_seq) for test_seq in TEST_CAFE_V]
    else:
        assert False

    PATHS = {
        "train": train_path,
        "val": val_path,
        "test": test_path
    }

    train_file = PATHS['train']
    val_file = PATHS['val']
    test_file = PATHS['test']

    return train_file, val_file, test_file


def remap_person_ids(persons):
    """
    persons: list of dicts with 'person_id'
    return: updated persons, old_id -> new_id map
    """
    old_ids = sorted({p["person_id"] for p in persons})
    id_map = {old_id: new_id+1 for new_id, old_id in enumerate(old_ids)}  # start by 1

    for p in persons:
        p["person_id"] = id_map[p["person_id"]]

    return persons, id_map

def remap_group_ids(groups, persons, person_id_map):
    """
    groups: list of dicts with 'group_id', 'include_id'
    """
    old_gids = sorted({g["group_id"] for g in groups})
    gid_map = {old_gid: new_gid+1 for new_gid, old_gid in enumerate(old_gids)}  # start by 1

    for g in groups:
        g["group_id"] = gid_map[g["group_id"]]
        g["include_id"] = [person_id_map[pid] for pid in g["include_id"]]
    for p in persons:
        p["group_id"] = gid_map[p["group_id"]]

    return persons, groups, gid_map


def cafe_read_annotations(ann_file):
    annotations = {}  # annotations for each frame
    action_name_to_id = {name: i for i, name in enumerate(Action_names)}
    activity_name_to_id = {name: i for i, name in enumerate(Activity_names)}
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        annotations = {}
        annotations['groups'] = []
        annotations['persons'] = []

        num_frames = data['framesCount']
        frame_interval = data['framesEach']
        actors = data['figures']
        key_frame = actors[0]['shapes'][0]['frame']

        for actor in actors:
            person_id = actor['id']
            group_name = actor['label']
            box = actor['shapes'][0]['coordinates']
            x1, y1 = box[0]
            x2, y2 = box[1]
            if x1 < 0: x1 = 0.0
            if y1 < 0: y1 = 0.0
            bbox = [x1, y1, x2, y2]
            if group_name != 'individual':  # group
                group_id = int(group_name[-1])  # start from 1

                if actor['attributes'][0]['value'] != "":
                    action = action_name_to_id[actor['attributes'][0]['value']['key']]
                    activity = action

                    if any(group.get('group_id') == group_id for group in annotations['groups']):
                        group = [group for group in annotations['groups'] if group.get('group_id') == group_id][0]
                        group['include_id'].append(person_id)
                    else:
                        annotations['groups'].append({
                                        'group_id': group_id,
                                        'activity': activity,
                                        'include_id': [person_id]
                                    })
                else:
                    if any(group.get('group_id') == group_id for group in annotations['groups']):
                        group = [group for group in annotations['groups'] if group.get('group_id') == group_id][0]
                        action = group['activity']
                    else:
                        action = -1

            else:  # individuals
                action = 6  # num_class + 1
                group_id = 0

            annotations['persons'].append({
                            'person_id': person_id,
                            'bbox': bbox,
                            'action': action,
                            'group_id': group_id
                        })

        for person in annotations['persons']:
            if person.get('action') == -1:
                person_id = person.get('person_id')
                group_id = person.get('group_id')
                if any(group.get('group_id') == group_id for group in annotations['groups']):
                    for group in annotations['groups']:
                        if group.get('group_id') == group_id:
                            person['action'] = group.get('activity')
                            group['include_id'].append(person_id)
                else:
                    person['action'] = 6
                    person['group_id'] = 0

        for person in annotations['persons']:
            if person.get('action') == 6:
                annotations['groups'].append({
                    'group_id': 0,
                    'activity': 6,
                    'include_id': [person.get('person_id')]
                })

    persons, person_id_map = remap_person_ids(annotations["persons"])
    persons, groups, group_id_map = remap_group_ids(annotations["groups"], persons, person_id_map)
    annotations["persons"] = persons
    annotations["groups"] = groups
    annotations['persons'].sort(key=lambda x: x['person_id'])
    annotations['groups'].sort(key=lambda x: x['group_id'])

    annotations['keyframe'] = int(key_frame)
    annotations['numframes'] = num_frames
    annotations['interval'] = frame_interval

    return annotations


def cafe_read_dataset(paths):
    data = {}
    for path in paths:
        sid = str(path).split('/')[-1]
        data[sid] = {}
        for cid in os.listdir(path):
            clip_path = os.path.join(path, cid)
            ann_file = clip_path + '/ann.json'
            data[sid][cid] = cafe_read_annotations(ann_file)  # data for each seq
    return data


'''
data structure:
data: {1: {ann1}, 2: {ann2}, ... , seq: {annseq}}
ann1(seq): {1: {ann1-1}, 2: {ann1-2}, ... , frame: {ann1-frame}}
ann1(seq)-1(frame): {persons: [{person_id: 1, bbox: [], action:, 1, group_id: 1}, ..., {...}], 
        groups: [{group_id: 1, activity: 1, person_ids: []}, ..., {...}]}
'''

def cafe_all_frames(anns, num_frames):
    half_left = num_frames // 2
    half_right = num_frames - half_left
    frames = []
    for s in anns:
        for f in anns[s]:
            if anns[s][f]['groups'] == []:
                continue
            else:
                frames.append((s, f, anns[s][f]['keyframe'], anns[s][f]['numframes'], anns[s][f]['interval']))
    return frames


class cafe_Dataset(data.Dataset):
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
        return len(self.frames)  # number of frames with anns (filtered the first and thea last ones)

    def __getitem__(self, idx):
        # Load feature map
        select_frames = self.get_frames(self.frames[idx])
        sample = self.load_samples_sequence(select_frames)

        return sample

    def get_frames(self, frame):
        sid, cid, kid, num, interval = frame  # s, f, anns[s][f]['keyframe'], anns[s][f]['numframes'], anns[s][f]['interval']

        # if self.is_training:
        #     half_left = self.num_frames // 2
        #     half_right = self.num_frames - half_left
        #
        #     fids = range(kid - half_left,
        #                  kid + half_right)
        #
        #     return sid, cid, [(sid, cid, fid) for fid in fids]
        # else:
        #     half_left = self.num_frames // 2
        #     half_right = self.num_frames - half_left
        #
        #     fids = range(kid - half_left,
        #                  kid + half_right)

        if self.is_training:
            # segment-based sampling
            segment_duration = num // self.num_frames
            sample_frames = np.multiply(list(range(self.num_frames)), segment_duration) + np.random.randint(
                segment_duration, size=self.num_frames)
        else:
            # segment-based sampling
            segment_duration = num // self.num_frames
            sample_frames = np.multiply(list(range(self.num_frames)), segment_duration) + np.random.randint(
                segment_duration, size=self.num_frames)

        return sid, cid, kid, [(sid, cid, int(fid * interval)) for fid in sample_frames]            # normal testing: each test loading 10 frames

    def load_samples_sequence(self, select_frames):
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda')
    #     else:
    #         device = torch.device('cpu')

        sid = select_frames[0]
        cid = select_frames[1]
        kid = select_frames[2]

        person_ids = []
        bboxes = []
        actions = []
        # p_group_ids = []

        group_ids = []
        activities = []
        include_ids = []

        for person in self.anns[sid][cid]['persons']:
            person_id = person['person_id']  # to connect the prediction and build pred matrix for iou group-level loss of group members
            person_ids.append(person_id)  # it is not in order in the tensor, following the order of group id, and some disappear
            bbox = person['bbox']  # for ROI Align and positional encoding
            bboxes.append(bbox)
            action = person['action']  # gt individual actions
            actions.append(action)
            # p_group_id = person['group_id']  # gt for person-level cross-entropy loss of group members
            # p_group_ids.append(p_group_id)

        for group in self.anns[sid][cid]['groups']:
            activity = group['activity']  # gt for group activity
            activities.append(activity)
            # group_id = group['group_id']
            # group_ids.append(group_id)
            include_id = group['include_id']
            include_ids.append(include_id)

        num_persons = len(person_ids)
        num_groups = len(activities)
        one_hot_matrix = np.zeros((num_persons, num_groups), dtype=float)
        # TODO: check if the order of column and the row should change
        person_to_index = {p: i for i, p in enumerate(person_ids)}

        for group, persons in enumerate(include_ids):
            for person in persons:
                one_hot_matrix[person_to_index[person], group] = 1


        imgs = []
        bbox = bboxes.copy()
        bbox = np.array(bbox, dtype=np.float64).reshape(-1, 4)
        for i, (sid, cid, fid) in enumerate(select_frames[3]):  # 10 frames for 1 item
            img = cv2.imread(self.img_path + '/' + sid + '/' + cid + '/images' + f'/frames_{fid}.jpg')[:, :, [2, 1, 0]]  # BGR -> RGB  # H, W, 3
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
        meta['cid'] = cid
        meta['kid'] = kid
        meta['frame_size'] = (1080, 1920)

        return imgs, bboxes, actions, activities, one_hot_matrix, meta