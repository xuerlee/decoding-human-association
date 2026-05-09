"""
Reference:
https://github.com/facebookresearch/detr
"""

from .model_Q import build


def build_model(args):
    return build(args)
