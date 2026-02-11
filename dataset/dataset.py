"""
Reference:
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark
"""
from pathlib import Path

import torch
import torch.utils.data as data
import torchvision

import util.transforms as visiontransforms

from .collective import collective_path, collective_read_dataset, collective_all_frames, Collective_Dataset
from .volleyball import volleyball_path, volleyball_read_dataset, volleyball_all_frames, Volleyball_Dataset
from .JRDB import jrdb_path, jrdb_read_dataset, jrdb_all_frames, jrdb_Dataset
from .cafe import cafe_path, cafe_read_dataset, cafe_all_frames, cafe_Dataset

def build(args):
    img_root = Path(args.img_path)
    assert img_root.exists(), f'provided image path {img_root} does not exist'
    ann_root = Path(args.ann_path)
    assert ann_root.exists(), f'provided bbox path {ann_root} does not exist'

    num_frames = args.num_frames


    if args.dataset == 'collective':
        train_ann_file, test_ann_file = collective_path(img_root, ann_root)

        train_anns = collective_read_dataset(train_ann_file)  # ann dictionary
        train_frames = collective_all_frames(train_anns, num_frames)  # frame and sec ids: (s, f)

        test_anns = collective_read_dataset(test_ann_file)
        test_frames = collective_all_frames(test_anns, num_frames)

        train_transform = visiontransforms.Compose([
        visiontransforms.RandomHorizontalFlip(),
        visiontransforms.Resize((args.img_h, args.img_w)),  # bbox resize is integrated in roialingn part
        # visiontransforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        visiontransforms.ToTensor(),  # PIL -> Tensor: HWC to CHW
        visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        test_transform = visiontransforms.Compose([
        visiontransforms.Resize((args.img_h, args.img_w)),
        visiontransforms.ToTensor(),
        visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_dataset = Collective_Dataset(train_anns, train_frames, args.img_path, train_transform,
                                          num_frames=args.num_frames, is_training=args.is_training)
        test_dataset = Collective_Dataset(test_anns, test_frames, args.img_path, test_transform,
                                         num_frames=args.num_frames, is_training=args.is_training)

        return train_dataset, test_dataset

    elif args.dataset == 'volleyball':
        train_ann_file, test_ann_file = volleyball_path(img_root, ann_root)

        train_anns = volleyball_read_dataset(train_ann_file, args)  # ann dictionary
        train_frames = volleyball_all_frames(train_anns, num_frames)  # frame and sec ids: (s, f)

        test_anns = volleyball_read_dataset(test_ann_file, args)
        test_frames = volleyball_all_frames(test_anns, num_frames)

        train_transform = visiontransforms.Compose([
        visiontransforms.RandomHorizontalFlip(),
        visiontransforms.Resize((args.img_h, args.img_w)),  # bbox resize is integrated in roialingn part
        # visiontransforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        visiontransforms.ToTensor(),  # PIL -> Tensor: HWC to CHW
        visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        test_transform = visiontransforms.Compose([
        visiontransforms.Resize((args.img_h, args.img_w)),
        visiontransforms.ToTensor(),
        visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_dataset = Volleyball_Dataset(train_anns, train_frames, args.img_path, train_transform,
                                          num_frames=args.num_frames, is_training=args.is_training)
        test_dataset = Volleyball_Dataset(test_anns, test_frames, args.img_path, test_transform,
                                         num_frames=args.num_frames, is_training=args.is_training)
        return train_dataset, test_dataset

    elif args.dataset == 'jrdb':
        train_ann_file, val_ann_file = jrdb_path(img_root, ann_root)

        train_anns = jrdb_read_dataset(args.jrdb_detection_path, train_ann_file)  # ann dictionary
        train_frames = jrdb_all_frames(train_anns, num_frames)  # frame and sec ids: (s, f)

        val_anns = jrdb_read_dataset(args.jrdb_detection_path, val_ann_file)
        val_frames = jrdb_all_frames(val_anns, num_frames)

        train_transform = visiontransforms.Compose([
        visiontransforms.RandomHorizontalFlip(),
        visiontransforms.Resize((args.img_h, args.img_w)),  # bbox resize is integrated in roialingn part
        # visiontransforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        visiontransforms.ToTensor(),  # PIL -> Tensor: HWC to CHW
        visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        val_transform = visiontransforms.Compose([
        visiontransforms.Resize((args.img_h, args.img_w)),
        visiontransforms.ToTensor(),
        visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_dataset = jrdb_Dataset(train_anns, train_frames, args.img_path, train_transform,
                                          num_frames=args.num_frames, is_training=args.is_training)
        val_dataset = jrdb_Dataset(val_anns, val_frames, args.img_path, val_transform,
                                         num_frames=args.num_frames, is_training=args.is_training)
        return train_dataset, val_dataset

    elif args.dataset == 'cafe':
        train_path, val_path, test_path = cafe_path(img_root, ann_root, args.cafe_split)

        train_anns = cafe_read_dataset(train_path)  # ann dictionary
        train_frames = cafe_all_frames(train_anns, num_frames)  # frame and sec ids: (s, f)

        val_anns = cafe_read_dataset(val_path)
        val_frames = cafe_all_frames(val_anns, num_frames)

        test_anns = cafe_read_dataset(test_path)
        test_frames = cafe_all_frames(test_anns, num_frames)

        train_transform = visiontransforms.Compose([
            visiontransforms.RandomHorizontalFlip(),
            visiontransforms.Resize((args.img_h, args.img_w)),  # bbox resize is integrated in roialingn part
            # visiontransforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            visiontransforms.ToTensor(),  # PIL -> Tensor: HWC to CHW
            visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        val_transform = visiontransforms.Compose([
            visiontransforms.Resize((args.img_h, args.img_w)),
            visiontransforms.ToTensor(),
            visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        test_transform = visiontransforms.Compose([
            visiontransforms.Resize((args.img_h, args.img_w)),
            visiontransforms.ToTensor(),
            visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_dataset = cafe_Dataset(train_anns, train_frames, args.img_path, train_transform,
                                          num_frames=args.num_frames, is_training=args.is_training)
        val_dataset = cafe_Dataset(val_anns, val_frames, args.img_path, val_transform,
                                    num_frames=args.num_frames, is_training=args.is_training)
        test_dataset = cafe_Dataset(test_anns, test_frames, args.img_path, test_transform,
                                         num_frames=args.num_frames, is_training=args.is_training)

        return train_dataset, val_dataset, test_dataset

    else:
        ValueError("Invalid dataset.")

