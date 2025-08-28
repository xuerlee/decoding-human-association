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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

@torch.no_grad()
def viz_i3d_feature_and_rois(  # to verify if roi bboxes are correct
    meta,
    action_fm: torch.Tensor,
    rois_source,                  # can be list[Tensor] (one rois per batch) or single Tensor (all samples concatenated together)
    B: int,
    T: int,
    sample_idx: int = 0,          # ith sample
    frame_idx: int = 0,           # tth frame
    persons="all",                # "all" | int(topk) | List[int]（idx for individuals）
    coords="feature",             # "feature" or "image"
    H: int = None, W: int = None, # coords="image"
    FH: int = None, FW: int = None,
    channel="mean",               # "mean" or idx of channels
    figsize=(5, 5),
    rect_kwargs=None              # rectangle style dict，ep: {"linewidth":2}
):
    """
    action_fm: (B*T, C, FH, FW) 的 I3D 特征
    rois_source:
        - list[Tensor]：every rois_i: (n_i*T, 5) = [bf_idx, x1, y1, x2, y2]
        - Tensor：(K, 5)，all roi concatenated together
    coords: rois coordinate system
        - "feature": rois coordinates on features
        - "image"  : rois coordinates on images
    persons:
        - "all"：draw all individuals
        - int  ：only draw topk individuals
        - List[int]： only draw certain individuals
    channel:
        - "mean"
        - int    ：only draw certain channels
    """
    assert action_fm.ndim == 4, f"action_fm shape should be (B*T, C, FH, FW), got {action_fm.shape}"
    _, C, FH0, FW0 = action_fm.shape
    FH = FH or FH0
    FW = FW or FW0

    # get the idx of this frame in action_fm
    bf_idx = sample_idx * T + frame_idx
    assert 0 <= bf_idx < action_fm.shape[0], f"bf_idx {bf_idx} out of range"

    # get frame feature
    feat = action_fm[bf_idx]  # (C, FH, FW)

    # visualization
    if channel == "mean":
        img = feat.mean(dim=0)
    else:
        assert isinstance(channel, int) and 0 <= channel < feat.shape[0], f"invalid channel={channel}"
        img = feat[channel]
    img = img.detach().float()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # collect the rois (bf_idx matching) for this frame
    if isinstance(rois_source, list):
        rois_all = rois_source[sample_idx]  # (n_i*T, 5)
        # in this sample，continuous T raws are corresponding to one individual；but can be also filtered by bf_idx
        sel = (rois_all[:, 0].long() == bf_idx)
        rois_this_frame = rois_all[sel]  # (n_t, 5)
    else:
        # single Tensor，directly filter bf_idx
        rois_all = rois_source
        sel = (rois_all[:, 0].long() == bf_idx)
        rois_this_frame = rois_all[sel]

    # select individuals to be drawn
    if persons == "all":
        pass
    elif isinstance(persons, int):
        rois_this_frame = rois_this_frame[:persons]
    else:
        # persons is idx list：every individual only has one bbox in one frame, get by idx directly
        rois_this_frame = rois_this_frame[persons]

    # transfer rois coordinate to feature coordinate
    if coords == "image":
        assert H is not None and W is not None, "coords='image' needs H,W"
        sx = FW / float(W)
        sy = FH / float(H)
        rois_feat = rois_this_frame.clone()
        rois_feat[:, [1, 3]] *= sx  # x1, x2
        rois_feat[:, [2, 4]] *= sy  # y1, y2
    else:
        rois_feat = rois_this_frame

    # draw
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img.cpu().numpy(), origin="upper")
    rk = {"fill": False, "linewidth": 2}
    if rect_kwargs:
        rk.update(rect_kwargs)

    # draw each boxes
    for r in rois_feat:
        _, x1, y1, x2, y2 = r.detach().cpu().tolist()

        x1 = max(0.0, min(float(x1), FW - 1))
        x2 = max(0.0, min(float(x2), FW - 1))
        y1 = max(0.0, min(float(y1), FH - 1))
        y2 = max(0.0, min(float(y2), FH - 1))
        if x2 > x1 and y2 > y1:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, **rk))

    ax.set_title(f"I3D feat | sid={meta['sid']}, fid={int(meta['src_fid'])}, frame={frame_idx}, bf_idx={bf_idx}")
    ax.set_xlim([0, FW])
    ax.set_ylim([FH, 0])  # y-axis is aheading down
    plt.tight_layout()
    plt.show()



class BackboneI3D(nn.Module):
    """
    RoI Align features for individuals for each frame in one batch;
    """

    def __init__(self, feature_channels, hidden_dim, crop_h, crop_w, n_blocks=3):
        super().__init__()
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.hidden_dim = hidden_dim
        self.i3d = i3d_noglobal(out_channel=hidden_dim)
        # self.i3d = i3d(out_channel=hidden_dim)
        self.roi_align = RoIAlign(output_size=(crop_h, crop_w), spatial_scale=1.0, sampling_ratio=-1)
        self.bbox_fc = nn.Sequential(nn.Linear(hidden_dim*crop_h*crop_w, 1024), nn.Linear(1024, hidden_dim))
        # self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, img, bbox, valid_areas_b, meta):
        B, T, C, H, W = img.shape
        # img.shape: 2, 10, 3, 224, 224; batch size, num_frames, C, H, W
        action_fm = self.i3d(img)
        # action_fm, global_fm = self.i3d(img)  # 20, 256, 14, 14  B*T,C,H,W
        _, C_o, FH, FW = action_fm.shape

        # remove the padded boxes before roi align
        all_rois = []
        bbox_copy = bbox.clone()
        n_max = 0  # the max number of persons in one frame in one batch
        n_per_frame = []
        # print(bbox_copy, valid_areas_b)
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
            rois = rois.repeat_interleave(T, dim=0)
            frame_id = (i * T) + torch.arange(0, T).repeat(n).reshape(-1, 1).to(rois.device)
            rois = torch.cat([frame_id, rois], dim=1)
            all_rois.append(rois)
        viz_i3d_feature_and_rois(
            meta[0],
            action_fm,
            all_rois,
            B=B, T=T,  # test when B = 1
            H=H, W=W,
            sample_idx=0,
            frame_idx=0,
            persons="all",
            coords="feature",
            channel="mean",
        )

        roi_boxes = torch.cat([b for i, b in enumerate(all_rois)], dim=0)  # N(all batch), 5  grouping boxes by individuals instead of frames
        # roi align
        boxes_idx_flat = roi_boxes[:, 0].long()  # N(all batch),
        # boxes_in_flat = roi_boxes[:, 1:]  # N(all batch), 4
        boxes_idx_flat.requires_grad = False
        roi_boxes.requires_grad = False

        boxes_features = self.roi_align(action_fm,
                                        roi_boxes)  # N (number of individuals in all batch with T frames per frame(stack together)), D(channels 256), K, K(crop size)

        # input proj (embeddings)
        # # Conv2D
        # boxes_features = self.input_proj(boxes_features)  # N(with T), 256, 7, 7
        # boxes_features = boxes_features.reshape(-1, T, self.hidden_dim, self.crop_h, self.crop_w)  # N(without T), T, hidden dim, crop size, crop size
        # boxes_features = boxes_features.flatten(2).permute(2, 0, 1)  # faltten from 2 dim to the last dim: 49, N, 256

        # FC (embeddings)
        N = boxes_features.shape[0]  # number of inviduals (with T)
        # boxes_features = torch.cat((boxes_features, global_fm), dim=0)  # only available when global fm is from mixed_5c
        # boxes_features = boxes_features.reshape(N+B*T, -1)
        boxes_features = boxes_features.reshape(N, -1)
        boxes_features = self.bbox_fc(boxes_features)

        # if not global features:
        boxes_features = boxes_features.reshape(-1, T, self.hidden_dim)  # since grouped bboxes by individuals instead of frames
        # add global features
        # global_features = global_fm.mean(dim=[2, 3])
        # boxes_features = torch.cat((boxes_features, global_features), dim=0).reshape(-1, T, self.hidden_dim)  # N(with T)+, T， hidden_dim(256)  calculate mean along T axis for the transformer output

        # padding and mask again
        start = 0
        boxes_features_padding = torch.zeros((B*n_max, T, self.hidden_dim), device=boxes_features.device)
        mask = torch.ones((B, T, n_max), dtype=torch.bool, device=boxes_features.device)
        for i, n in enumerate(n_per_frame):
            boxes_features_padding[i*n_max: i*n_max+n, :, :].copy_(boxes_features[start: start+n, :, :])
            mask[i, :, :n] = False
            start += n
        boxes_features_padding = boxes_features_padding.reshape(B, n_max, T, self.hidden_dim).permute(0, 2, 1, 3).reshape(B*T, n_max, self.hidden_dim).permute(1, 0, 2)
        # boxes_features_padding = boxes_features_padding.reshape(B, n_max, T, self.hidden_dim).permute(0, 2, 1, 3).reshape(B*T, n_max, self.hidden_dim)
        mask = mask.reshape(B*T, n_max)  # n_max, B*T, hidden_dim, find connections between individuals per frame
        # mask = mask.reshape(B*T, n_max).permute(1, 0)
        # print(mask)
        return roi_boxes, boxes_features_padding, mask, n_max, n_per_frame, (FH, FW)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.T = position_embedding.T

    def forward(self, fm, bbox, valid_areas_b, meta):
        # print(meta)
        roi_boxes, boxes_features, mask, n_max, n_per_frame, featuremap_size = self[0](fm, bbox, valid_areas_b, meta)  # backbone

        bbox_norm = roi_boxes.clone()
        start = 0
        for i, n in enumerate(n_per_frame):
            bbox_norm[start*self.T: start*self.T+n*self.T, [1, 3]] /= featuremap_size[1]
            bbox_norm[start*self.T: start*self.T+n*self.T, [2, 4]] /= featuremap_size[0]
            start += n
        bbox_norm = bbox_norm[:, 1:]
        pos = self[1](bbox_norm, n_max, n_per_frame)

        return boxes_features, pos, mask


def build_backboneI3D(args):
    position_embedding = build_position_encoding(args)
    backbone = BackboneI3D(args.feature_channels, args.hidden_dim, args.roi_align[0], args.roi_align[1])
    model = Joiner(backbone, position_embedding)
    return model
