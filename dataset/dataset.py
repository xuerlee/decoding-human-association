from pathlib import Path

import torch
import torch.utils.data as data
import torchvision

from .collective import collective_path, collective_read_dataset, collective_all_frames, Collective_Dataset

def build(args):
    img_root = Path(args.img_path)
    assert img_root.exists(), f'provided image path {img_root} does not exist'
    ann_root = Path(args.ann_path)
    assert ann_root.exists(), f'provided bbox path {ann_root} does not exist'

    if args.feature_file == 'collective':
        train_img_file, train_ann_file, test_img_file, test_ann_file = collective_path(img_root, ann_root)

        train_anns = collective_read_dataset(train_ann_file)  # ann dictionary
        train_frames = collective_all_frames(train_anns)  # frame and sec ids: (s, f)
        # print(train_frames)

        test_anns = collective_read_dataset(test_ann_file)
        test_frames = collective_all_frames(test_anns)

        train_dataset = Collective_Dataset(train_anns, train_frames, args.img_path, args.img_w, args.img_h,
                                          num_frames=args.num_frames, is_training=args.is_training)
        test_dataset = Collective_Dataset(test_anns, test_frames, args.img_path, args.img_w, args.img_h,
                                         num_frames=args.num_frames, is_training=args.is_training)

    else:
        ValueError("Invalid dataset.")

    return train_dataset, test_dataset

