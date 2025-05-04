import torch
import torch.utils.data
import torchvision

from .featuremap import build

def build_fmset(args):
    if args.feature_map_path:
        return build(args)
    raise ValueError(f'feature map path {args.feature_map_path} not supported')
