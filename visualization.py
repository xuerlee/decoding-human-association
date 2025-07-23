"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import random
import torch
import util.misc as utils
import colorsys
import cv2
from torchvision import transforms

action_names = ['none', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']
activity_names = ['none', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking', 'Empty']

class DistinctColorGenerator:
    def __init__(self, saturation=0.7, value=0.9):
        self.index = 0
        self.s = saturation
        self.v = value

    def next(self):
        h = (self.index * 0.61803398875) % 1  # golden interval angles
        r, g, b = colorsys.hsv_to_rgb(h, self.s, self.v)
        self.index += 1
        return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV

def load_key_img(img_folder_path, meta):
    key_imgs = []
    sids = []
    fids = []
    for item in meta:
        sid = str(item['sid']).zfill(2)
        fid = str(item['src_fid']).zfill(4)
        img_path = img_folder_path + '/seq' + sid + '/frame' + fid + '.jpg'
        key_img = cv2.imread(img_path)
        key_img = cv2.cvtColor(key_img, cv2.COLOR_BGR2RGB)
        key_imgs.append(key_img)
        sids.append(sid)
        fids.append(fid)
    return key_imgs, sids, fids
def merge_group_bboxes(bboxes: torch.Tensor, group_ids: torch.Tensor):
    group_bboxes = []
    unique_groups = torch.unique(group_ids)
    person_ids = []
    for gid in unique_groups:
        group_mask = group_ids == gid  # person ids mask
        person_id = torch.where(group_mask==True)[0]
        person_ids.append(person_id)
        group_boxes = bboxes[group_mask]

        x1 = group_boxes[:, 0].min()
        y1 = group_boxes[:, 1].min()
        x2 = group_boxes[:, 2].max()
        y2 = group_boxes[:, 3].max()

        group_bboxes.append([x1.item(), y1.item(), x2.item(), y2.item()])

    return group_bboxes, person_ids

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))  # BGR

def draw_bboxes(img, bboxes, action_labels, group_bboxes, person_ids, activity_labels):
    color_gen = DistinctColorGenerator()
    img_copy = img.copy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(person_ids, torch.Tensor):
        person_ids = person_ids.cpu().numpy()

    for i, group_bbox in enumerate(group_bboxes):
        x1, y1, x2, y2 = map(int, group_bbox)
        color = color_gen.next()
        cv2.rectangle(img_copy, (x1-17, y1-17), (x2+17, y2+17), color, thickness=2)
        activity_label = activity_names[activity_labels[i]]
        cv2.putText(img_copy, activity_label, (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

        person_id = person_ids[i]

        for pid in person_id:
            x1, y1, x2, y2 = map(int, bboxes[pid])
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            action_label = action_names[action_labels[pid]]
            cv2.putText(img_copy, action_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    return img_copy

def draw_bboxes_compare(img, bboxes, action_labels, action_gts):
    color_gen = DistinctColorGenerator()
    img_copy = img.copy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        color = color_gen.next()
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness=2)

        action_label = action_names[action_labels[i]]
        cv2.putText(img_copy, action_label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        action_gt = action_names[action_gts[i]]
        cv2.putText(img_copy, action_gt, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    return img_copy

@torch.no_grad()
def visualization(model, criterion, data_loader, device, args):
    model.eval()
    criterion.eval()
    img_folder_path = args.img_path
    output_dir = args.output_dir

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Output:'

    for iteration, (samples, targets, meta) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)  # feature maps
        targets = [t.to(device) for t in targets]  # bboxes, actions, activities, one_hot_matrix

        key_imgs, sids, fids = load_key_img(img_folder_path, meta)  # numpy
        bboxes = targets[0].decompose()[0]
        action_gts = targets[1].decompose()[0]
        outputs = model(samples, targets[0], meta)

        pred_action_logits = outputs['pred_action_logits']
        pred_activity_logits = outputs['pred_activity_logits']
        attention_weights = outputs['attention_weights']  # B, num_max_person, num_queries

        group_bboxes = []
        person_ids = []
        valid_activity_labels = []
        for i, aw in enumerate(attention_weights):
            aw = aw[~(aw == 0).all(dim=1)]
            group_ids_person = aw.argmax(dim=-1)
            bbox = bboxes[i][~(bboxes[i] == 0).all(dim=1)]
            group_bbox, person_id = merge_group_bboxes(bbox, group_ids_person)
            group_bboxes.append(group_bbox)
            person_ids.append(person_id)

            pred_activity_logit = pred_activity_logits[i][~(pred_activity_logits[i] == 0).all(dim=1)]
            activity_labels = pred_activity_logit.argmax(dim=-1)
            unique_groups = torch.unique(group_ids_person)
            valid_activity_label = activity_labels[unique_groups]
            valid_activity_labels.append(valid_activity_label)

        for i, key_img in enumerate(key_imgs):
            bbox = bboxes[i][~(bboxes[i] == 0).all(dim=1)]
            pred_action_logit = pred_action_logits[i][~(pred_action_logits[i] == 0).all(dim=1)]
            action_labels = pred_action_logit.argmax(dim=-1)
            # img_with_bbox = draw_bboxes(key_img, bbox, action_labels, group_bboxes[i], person_ids[i], valid_activity_labels[i])
            img_with_bbox = draw_bboxes_compare(key_img, bbox, action_labels, action_gts[i])
            img_with_bbox = cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR)
            # cv2.imshow(f'img_seq{sids[i]}_frame{fids[i]}', img_with_bbox)
            # cv2.waitKey(0)
            cv2.imwrite(output_dir+f'/img_seq{sids[i]}_frame{fids[i]}.jpg', img_with_bbox)
    return
