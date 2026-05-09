"""
Reference:
https://github.com/facebookresearch/detr
"""

from .model import build


def build_model(args):
    return build(args)
