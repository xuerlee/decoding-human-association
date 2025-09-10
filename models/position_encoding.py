# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn



class PositionEmbeddingLearned(nn.Module):
    """
    learned pos embedding based on bboxes relative position
    """
    def __init__(self, B, T, hidden_dim=256):
        super().__init__()
        self.pos_fc = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, hidden_dim))
        self.hidden_dim = hidden_dim
    def forward(self, bbox, n_max, n_per_frame, T_o):  # bbox: N(with T), 4
        B = len(n_per_frame)  # in evaluation last batch (no enough samples for one batch)
        pos = self.pos_fc(bbox).reshape(-1, T_o, self.hidden_dim)  # N(without T), T, hidden_dim(256)
        # pos = torch.unsqueeze(pos, 0)  # 1, N, 256
        pos_pad = torch.zeros((B * n_max, T_o, self.hidden_dim), device=pos.device)
        start = 0
        for i, n in enumerate(n_per_frame):
            pos_pad[i*n_max: i*n_max+n, :, :].copy_(pos[start: start+n, :, :])
            start += n
        pos_pad = pos_pad.reshape(B, n_max, T_o, self.hidden_dim).permute(0, 2, 1, 3).reshape(B*T_o, n_max, self.hidden_dim).permute(1, 0, 2).contiguous()
        return pos_pad


def build_position_encoding(args):
    position_embedding = PositionEmbeddingLearned(args.batch_size, args.num_frames, args.hidden_dim)

    return position_embedding
