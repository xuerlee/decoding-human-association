"""
Reference:
https://github.com/facebookresearch/detr
"""

from .detr import build


def build_model(args):
    return build(args)
