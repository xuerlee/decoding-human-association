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
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['PYTHONMALLOC'] = 'debug'
# os.environ['MALLOC_CHECK_'] = '3'
# os.environ['MALLOC_PERTURB_'] = '153'

from featuremaps import build_fmset
from dataset import build_dataset

import util.misc as utils
from engine import evaluate, train_one_epoch, train_one_epoch_accum_steps
from models import build_model



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--if_accum', default=False, action='store_true')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=[50, 100], nargs='+', type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Transformer
    parser.add_argument('--enc_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
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

    # Loss
    parser.add_argument('--aux_loss', default=False, type=bool,
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_activity_class', default=2, type=float,
                        help="Group activity class coefficient in the matching cost")
    parser.add_argument('--set_cost_action_class', default=1, type=float,
                        help="Individual action consistence coefficient in the matching cost")
    parser.add_argument('--set_cost_bce', default=5, type=float,
                        help="BCE error between one-hot grouping matrices and cross attention weights coefficient in the matching cost")
    parser.add_argument('--set_cost_size', default=5, type=float,
                        help="L1 cost between one-hot grouping matrices and cross attention weights coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--action_loss_coef', default=2, type=float)
    parser.add_argument('--activity_loss_coef', default=2, type=float)
    parser.add_argument('--grouping_loss_coef', default=3, type=float)
    parser.add_argument('--consistency_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object classes (empty groups)")

    # feature map preparing & roi align
    parser.add_argument('--dataset',
                        # default='collective',
                        # default='volleyball',
                        # default='jrdb',
                        default='cafe',
                        help='choose the dataset: collective, volleyball, jrdb, cafe')
    parser.add_argument('--cafe_split',
                        default='view',
                        help='place or view')
    parser.add_argument('--input_format', default='image',
                        help='choose original images or extracted features in numpy format: image or feature')
    parser.add_argument('--feature_map_path',
                        default='/home/jiqqi/data/new-new-collective/img_for_fm_fm', type=str)
    parser.add_argument('--img_path',
                        # default='/home/jiqqi/data/new-new-collective/ActivityDataset',
                        # default='/media/jiqqi/新加卷/dataset/volleyball_/videos',
                        # default='/media/jiqqi/新加卷/dataset/JRDB/train_images/images',
                        default='/media/jiqqi/OS/dataset/Cafe_Dataset/Dataset/cafe',
                        type=str)
    parser.add_argument('--ann_path',
                        # default='/home/jiqqi/data/social_CAD/anns',
                        # default='/home/jiqqi/data/Volleyball/volleyball_tracking_annotation',
                        # default='/media/jiqqi/新加卷/dataset/JRDB/train_images/labels/labels_2d',
                        default='/media/jiqqi/OS/dataset/Cafe_Dataset/evaluation/gt_tracks.txt',
                        type=str)
    parser.add_argument('--is_training', default=True, type=bool,
                        help='data preparation may have differences')
    parser.add_argument('--img_w', default=1280, type=int,
                        help='width of resized images')
    parser.add_argument('--img_h', default=720, type=int,
                        help='heigh of resized images')
    parser.add_argument('--num_frames', default=1, type=int,
                        help='number of stacked frame features')
    parser.add_argument('--feature_channels', default=1392, type=int,  # openpifpaf output
                        help='number of feature channels output by the feature extraction part')
    parser.add_argument('--roi_align', default=[7, 7], type=int,  # openpifpaf output
                        help='size of roi_align')

    parser.add_argument('--output_dir', default='output_dir/test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--runs_dir', default='runs/test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume',
                        default='',
                        # default='output_dir/restartall_hidden256_enc2dec2_cafe_kinetics_sampleequal_CyclicLR/checkpoint0004.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # writer_dir = args.output_dir.split('/')[-1]
    if not args.eval:
        # writer = SummaryWriter(log_dir=f'runs/{writer_dir}')
        writer = SummaryWriter(log_dir=args.runs_dir)
    else:
        writer = None
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
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    param_dicts = {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]},

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-4, step_size_up=4,
                                                  step_size_down=25, mode='triangular2',
                                                  cycle_momentum=False)

    # optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=0.0000001)
    # lr_scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)


    if args.input_format == 'feature':  # only collective is available
        dataset_train, dataset_val = build_fmset(args=args)
    elif args.input_format == 'image':
        if args.dataset == 'cafe':
            dataset_train, dataset_val, dataset_test = build_dataset(args=args)
        else:
            dataset_train, dataset_val = build_dataset(args=args)
    else:
        raise ValueError(f'import format {args.input_format} not supported, options: image or feature')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if args.dataset == 'cafe':
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # only shuffle with the clips, not the frames inside the clips
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if args.dataset == 'cafe':
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)  # collate_fn return: CUDA tensor and mask
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, collate_fn=utils.collate_fn,
                                 drop_last=False, num_workers=args.num_workers)
    if args.dataset == 'cafe':
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, collate_fn=utils.collate_fn,
                                     drop_last=False, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    save_path = args.resume.split('/checkpoint')[0]
    if args.eval:
        if args.dataset == 'cafe':
            test_stats = evaluate(args, args.dataset, model, criterion, data_loader_test, device, save_path, if_confuse=True)
            print('test stats:', test_stats)
            return
        else:
            test_stats = evaluate(args, args.dataset, model, criterion, data_loader_val, device, save_path, if_confuse=True)
            print('test stats:', test_stats)
            return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if args.if_accum:
            train_stats = train_one_epoch_accum_steps(
                model, criterion, data_loader_train, optimizer, device, epoch, writer, args.accum_steps,
                args.clip_max_norm)  # engine.py -- train_one_epoch
        else:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch, writer,
                args.clip_max_norm)  # engine.py -- train_one_epoch
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            # for epoch_lr in args.lr_drop:
                # if (epoch + 1) % epoch_lr == 0 or (epoch + 1) % 100 == 0:
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(args, args.dataset,
            model, criterion, data_loader_val, device, save_path, if_confuse=False
        )

        if writer is not None:
            for k, v in test_stats.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"Test/{k}", v, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
