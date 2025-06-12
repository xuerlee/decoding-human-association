# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
prepare for the input for transformer
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import RoIAlign
from typing import Dict, List
from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class Backbone(nn.Module):
    """
    RoI Align features for individuals for each frame in one batch;
    """

    def __init__(self, feature_channels, hidden_dim, crop_h, crop_w):
        super().__init__()
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.hidden_dim = hidden_dim
        self.roi_align = RoIAlign(output_size=(crop_h, crop_w), spatial_scale=1.0, sampling_ratio=-1)
        self.bbox_fc = nn.Sequential(nn.Linear(feature_channels*crop_h*crop_w, 1024), nn.Linear(1024, hidden_dim))
        self.input_proj = nn.Conv2d(feature_channels, hidden_dim, kernel_size=1)
    def forward(self, fm, bbox, valid_areas_b, meta):
        B, T, C, H, W = fm.shape
        # fm.shape: 2, 10, 1392, 31, 46 batch size, num_frames, C, H, W
        # remove the padded boxes before roi align
        all_rois = []
        bbox_copy = bbox.clone()
        n_max = 0  # the max number of persons in one frame in one batch
        n_per_frame = []
        for i, bbox_b in enumerate(bbox_copy):
            rois = bbox_b[valid_areas_b[i][0]: valid_areas_b[i][1], valid_areas_b[i][2]: valid_areas_b[i][3]]
            H_ratio = meta[0]['down_sample_ratio'][0]
            W_ratio = meta[0]['down_sample_ratio'][1]
            rois[:, [0, 2]] *= W_ratio
            rois[:, [1, 3]] *= H_ratio  # : represents selecting all rows
            n = rois.shape[0]
            n_per_frame.append(n)
            if n > n_max:
                n_max = n
            rois = rois.repeat_interleave(T, dim=0)
            frame_id = (i * T) + torch.arange(0, T).repeat(n).reshape(-1, 1).to(rois.device)
            rois = torch.cat([frame_id, rois], dim=1)
            all_rois.append(rois)

        roi_boxes = torch.cat([b for i, b in enumerate(all_rois)], dim=0)  # N(all batch), 5
        # roi align
        boxes_idx_flat = roi_boxes[:, 0].long()  # N(all batch),
        # boxes_in_flat = roi_boxes[:, 1:]  # N(all batch), 4
        boxes_idx_flat.requires_grad = False
        roi_boxes.requires_grad = False

        fm = fm.reshape(B*T, C, H, W)

        boxes_features = self.roi_align(fm,
                                        roi_boxes)  # N (number of individuals in all batch with T frames per frame), D(channels 1392), K, K(crop size)

        # input proj (embeddings)
        # # Conv2D
        # boxes_features = self.input_proj(boxes_features)  # N(with T), 256, 7, 7
        # boxes_features = boxes_features.reshape(-1, T, self.hidden_dim, self.crop_h, self.crop_w)  # N(without T), T, hidden dim, crop size, crop size
        # boxes_features = boxes_features.flatten(2).permute(2, 0, 1)  # faltten from 2 dim to the last dim: 49, N, 256
        # FC
        N = boxes_features.shape[0]  # number of inviduals (with T)
        boxes_features = boxes_features.reshape(N, -1)
        boxes_features = self.bbox_fc(boxes_features).reshape(-1, T, self.hidden_dim)  # N(with T), T， hidden_dim(256)  calculate mean along T axis for the transformer output

        # padding and mask again
        start = 0
        boxes_features_padding = torch.zeros((B*n_max, T, self.hidden_dim), device=boxes_features.device)
        mask = torch.ones((B, T, n_max), dtype=torch.bool, device=boxes_features.device)
        for i, n in enumerate(n_per_frame):
            boxes_features_padding[i*n_max: i*n_max+n, :, :].copy_(boxes_features[start: start+n, :, :])
            mask[i, :, :n] = False
            start += n
        boxes_features_padding = boxes_features_padding.reshape(B, n_max, T, self.hidden_dim).permute(0, 2, 1, 3).reshape(B*T, n_max, self.hidden_dim).permute(1, 0, 2)
        mask = mask.reshape(B*T, n_max)  # n_max, B*T, hidden_dim, find connections between individuals per frame
        return roi_boxes, boxes_features_padding, mask, n_max, n_per_frame


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.T = position_embedding.T

    def forward(self, fm, bbox, valid_areas_b, meta):
        # print(meta)
        roi_boxes, boxes_features, mask, n_max, n_per_frame = self[0](fm, bbox, valid_areas_b, meta)  # backbone

        bbox_norm = roi_boxes.clone()
        start = 0
        for i, n in enumerate(n_per_frame):
            featuremap_size = meta[i]['featuremap_size']
            bbox_norm[start*self.T: start*self.T+n*self.T, [1, 3]] /= featuremap_size[1]
            bbox_norm[start*self.T: start*self.T+n*self.T, [2, 4]] /= featuremap_size[0]
            start += n
        bbox_norm = bbox_norm[:, 1:]
        pos = self[1](bbox_norm, n_max, n_per_frame)

        return boxes_features, pos, mask


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(args.feature_channels, args.hidden_dim, args.roi_align[0], args.roi_align[1])
    model = Joiner(backbone, position_embedding)
    return model


# TODO: DELETE backbone and replace it by roi align (maybe at the detr class), rewrite positional encodding part