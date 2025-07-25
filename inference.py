import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import featuremaps
from featuremaps import build_fmset
from dataset import build_dataset

import util.misc as utils
from engine import evaluate, train_one_epoch
from visualization import visualization
from models import build_model



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots (for group prediction)")
    parser.add_argument('--pre_norm', action='store_true')  # layer norm (similar to batch norm, normalize in each input tensor)

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--action_loss_coef', default=2, type=float)
    parser.add_argument('--activity_loss_coef', default=2, type=float)
    parser.add_argument('--grouping_loss_coef', default=5, type=float)
    parser.add_argument('--consistency_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object classes (empty groups)")

    # * Matcher
    parser.add_argument('--set_cost_activity_class', default=2, type=float,
                        help="Group activity class coefficient in the matching cost")
    parser.add_argument('--set_cost_action_class', default=1, type=float,
                        help="Individual action consistence coefficient in the matching cost")
    parser.add_argument('--set_cost_bce', default=5, type=float,
                        help="BCE error between one-hot grouping matrices and cross attention weights coefficient in the matching cost")

    # feature map preparing & roi align
    parser.add_argument('--feature_file', default='collective',
                        help='choose the dataset: collective or volleyball')
    parser.add_argument('--input_format', default='image',
                        help='choose original images or extracted features in numpy format: image or feature')
    parser.add_argument('--feature_map_path',
                        default='/home/jiqqi/data/new-new-collective/img_for_fm_fm', type=str)
    parser.add_argument('--img_path',
                        default='/home/jiqqi/data/new-new-collective/ActivityDataset', type=str)
    parser.add_argument('--ann_path',
                        default='/home/jiqqi/data/social_CAD/anns', type=str)
    parser.add_argument('--is_training', default=True, type=bool,
                        help='data preparation may have differences')
    parser.add_argument('--img_w', default=224, type=int,
                        help='width of resized images')
    parser.add_argument('--img_h', default=224, type=int,
                        help='heigh of resized images')
    parser.add_argument('--num_frames', default=10, type=int,
                        help='number of stacked frame features')
    parser.add_argument('--feature_channels', default=1392, type=int,  # openpifpaf output
                        help='number of feature channels output by the feature extraction part')
    parser.add_argument('--roi_align', default=[7, 7], type=int,  # openpifpaf output
                        help='size of roi_align')

    parser.add_argument('--output_dir', default='output_imgs/i3d/i3d_action_enc6_4f/test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume',
                        # default='',
                        default='output_dir/i3d/i3d_action_enc6_4f/checkpoint0299.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=True, action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train, dataset_val = build_fmset(args=args)

    if args.input_format == 'feature':
        dataset_train, dataset_val = build_fmset(args=args)
    elif args.input_format == 'image':
        dataset_train, dataset_val = build_dataset(args=args)
    else:
        raise ValueError(f'import format {args.input_format} not supported, options: image or feature')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # only shuffle with the clips, not the frames inside the clips
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)  # collate_fn return: CUDA tensor and mask
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, collate_fn=utils.collate_fn,
                                 drop_last=False, num_workers=args.num_workers)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    print("Start inference")
    start_time = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    visualization(model, criterion, data_loader_val, device, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Inference time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
