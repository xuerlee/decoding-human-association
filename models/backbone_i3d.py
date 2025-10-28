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

from .i3d import i3d, i3d_noglobal
from .position_encoding import build_position_encoding




class BackboneI3D(nn.Module):
    """
    RoI Align features for individuals for each frame in one batch;
    """

    def __init__(self, feature_channels, hidden_dim, crop_h, crop_w):
        super().__init__()
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.hidden_dim = hidden_dim
        # self.hidden_dim = 256*7*7
        self.i3d = i3d_noglobal(out_channel=hidden_dim)
        # self.i3d = i3d(out_channel=hidden_dim)
        self.roi_align = RoIAlign(output_size=(crop_h, crop_w), spatial_scale=1, sampling_ratio=-1)
        # self.bbox_conv = nn.Sequential(
        #     nn.Conv2d(hidden_dim, 64, kernel_size=1),
        #     nn.ReLU(inplace=True))
        self.bbox_fc = nn.Sequential(nn.Linear(hidden_dim*crop_h*crop_w, 1024), nn.Linear(1024, hidden_dim))

    def forward(self, img, bbox, valid_areas_b, meta):
        B, _, C, H, W = img.shape  # img.shape: 2, 10, 3, 224, 224; batch size, num_frames, C, H, W
        action_fm = self.i3d(img).contiguous()
        # action_fm, global_fm = self.i3d(img)  # B, 256, 14, 14  B,C,H,W
        _, C_o, FH, FW = action_fm.shape

        # remove the padded boxes before roi align
        all_rois = []
        bbox_copy = bbox.clone()
        n_max = 0  # the max number of persons in one frame in one batch
        n_per_frame = []
        for i, bbox_b in enumerate(bbox_copy):
            rois = bbox_b[valid_areas_b[i][0]: valid_areas_b[i][1], valid_areas_b[i][2]: valid_areas_b[i][3]]
            # the bboxes has been scaled once in Transforms
            OH = meta[0]['frame_size'][0]
            OW = meta[0]['frame_size'][1]
            H_ratio = FH / OH
            W_ratio = FW / OW
            rois[:, [0, 2]] *= W_ratio
            rois[:, [1, 3]] *= H_ratio  # : represents selecting all rows
            n = rois.shape[0]
            n_per_frame.append(n)
            if n > n_max:
                n_max = n  # n_max in one batch
            frame_id = torch.tensor(i).repeat(n).reshape(-1, 1).to(rois.device)
            rois = torch.cat([frame_id, rois], dim=1)
            all_rois.append(rois)


        roi_boxes = torch.cat([b for i, b in enumerate(all_rois)], dim=0)  # N(all batch), 5  grouping boxes by individuals instead of frames
        # roi align
        boxes_idx_flat = roi_boxes[:, 0].long()  # N(all batch),
        # boxes_in_flat = roi_boxes[:, 1:]  # N(all batch), 4
        boxes_idx_flat.requires_grad = False
        roi_boxes.requires_grad = False

        boxes_features_ini = self.roi_align(action_fm,
                                        roi_boxes)  # N (number of individuals in all batch with T_o frames per frame(stack together)), D(channels 256), K, K(crop size)

        # FC (embeddings)
        N = boxes_features_ini.shape[0]  # number of inviduals
        # boxes_features = torch.cat((boxes_features, global_fm), dim=0)  # only available when global fm is from mixed_5c
        # boxes_features = boxes_features.reshape(N+B, -1)

        # *******************************************
        # boxes_features = self.bbox_conv(boxes_features)
        boxes_features = boxes_features_ini.reshape(N, -1)
        boxes_features = self.bbox_fc(boxes_features)
        boxes_features = boxes_features.reshape(N, self.hidden_dim).contiguous()  # since grouped bboxes by individuals instead of frames
        # ********************************************

        # if add global features ()
        # TODO: 1) cat global features (B, 256) to boxes features (N, 256) or 2) add global tokens and adjust padding boxes features and mask
        # global_features = global_fm.mean(dim=[2, 3])  # FH, FW
        # boxes_features = torch.cat((boxes_features, global_features), dim=0)

        # padding and mask again
        start = 0
        boxes_features_padding = torch.zeros((B*n_max, self.hidden_dim), device=boxes_features.device)
        mask = torch.ones((B, n_max), dtype=torch.bool, device=boxes_features.device)
        for i, n in enumerate(n_per_frame):
            boxes_features_padding[i*n_max: i*n_max+n, :].copy_(boxes_features[start: start+n, :])
            mask[i, :n] = False
            start += n
        boxes_features_padding = boxes_features_padding.reshape(B, n_max, self.hidden_dim).contiguous()  # B, n_max, hidden_dim
        mask = mask.reshape(B, n_max)
        boxes_features_padding = boxes_features_padding.permute(1, 0, 2).contiguous()  # n_max, B, hidden_dim

        start = 0
        boxes_features_ini_padding = torch.zeros((B * n_max, self.hidden_dim * self.crop_h * self.crop_w), device=boxes_features_ini.device)
        for i, n in enumerate(n_per_frame):
            boxes_features_ini_padding[i * n_max: i * n_max + n, :].copy_(boxes_features_ini[start: start + n, :])
            start += n
        boxes_features_ini_padding = boxes_features_ini_padding.reshape(B, n_max,
                                                                self.hidden_dim * self.crop_h * self.crop_w).contiguous()  # B, n_max, hidden_dim*roi_w*roi_h
        boxes_features_ini_padding = boxes_features_ini_padding.permute(1, 0, 2).contiguous()  # n_max, B, hidden_dim

        return roi_boxes, boxes_features_padding, boxes_features_ini_padding, mask, n_max, n_per_frame, (FH, FW)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, fm, bbox, valid_areas_b, meta):
        # print(meta)
        roi_boxes, boxes_features, boxes_features_ini, mask, n_max, n_per_frame, featuremap_size = self[0](fm, bbox, valid_areas_b, meta)  # backbone

        bbox_norm = roi_boxes.clone()
        start = 0
        for i, n in enumerate(n_per_frame):
            bbox_norm[start: start+n, [1, 3]] /= featuremap_size[1]
            bbox_norm[start: start+n, [2, 4]] /= featuremap_size[0]
            start += n
        bbox_norm = bbox_norm[:, 1:]
        pos = self[1](bbox_norm, n_max, n_per_frame)

        return boxes_features, boxes_features_ini, pos, mask


def build_backboneI3D(args):
    position_embedding = build_position_encoding(args)
    backbone = BackboneI3D(args.feature_channels, args.hidden_dim, args.roi_align[0], args.roi_align[1])
    model = Joiner(backbone, position_embedding)
    return model
