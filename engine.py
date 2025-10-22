"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

action_names = ['none', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']
activity_names = ['none', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking', 'Empty']


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('activity_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('action_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for iteration, (samples, targets, meta) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)  # feature maps
        targets = [t.to(device) for t in targets]  # bboxes, actions, activities, one_hot_matrix
        outputs = model(samples, targets[0], meta)
        loss_dict = criterion(outputs, targets[1:])
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all
        # GPUs for logging purposes (average loss of all GPUs)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}  # losses without multiplying weights
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(activity_class_error=loss_dict_reduced['activity_class_error'])
        # metric_logger.update(activity_class_error=0)
        metric_logger.update(action_class_error=loss_dict_reduced['action_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None:
            global_step = epoch * len(data_loader) + iteration
            writer.add_scalar('Loss/total', loss_value, global_step)

            for k, v in loss_dict_reduced_scaled.items():
                if k.startswith('action_class_error_') or k.startswith('activity_class_error_'):
                    writer.add_scalar(f'Loss_scaled/{k}', v, global_step)
                else:
                    writer.add_scalar(f'Loss_scaled/{k}', v.item(), global_step)

            for k, v in loss_dict_reduced_unscaled.items():
                if k.startswith('action_class_error_') or k.startswith('activity_class_error_'):
                    writer.add_scalar(f'Loss_unscaled/{k}', v, global_step)
                else:
                    writer.add_scalar(f'Loss_unscaled/{k}', v.item(), global_step)

            # writer.add_scalar('Error/activity_class_error', loss_dict_reduced['activity_class_error'], global_step)
            writer.add_scalar('Error/action_class_error', loss_dict_reduced['action_class_error'], global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]["lr"], global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, save_path, if_confuse=False):
    model.eval()
    criterion.eval()

    all_action_preds = []
    all_action_gts = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('activity_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('action_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    for iteration, (samples, targets, meta) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)  # feature maps
        targets = [t.to(device) for t in targets]  # bboxes, actions, activities, one_hot_matrix
        outputs = model(samples, targets[0], meta)
        loss_dict = criterion(outputs, targets[1:])
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        # TODO: grouping error
        # metric_logger.update(activity_class_error=loss_dict_reduced['activity_class_error'])
        metric_logger.update(action_class_error=loss_dict_reduced['action_class_error'])
        for k, v in loss_dict_reduced.items():
            if k.startswith('action_class_error_') or k.startswith('activity_class_error_') and v is not None:
                metric_logger.update(**{k: v})

        # for confusion matrix
        if if_confuse:
            pred_action_logits = outputs['pred_action_logits']
            action_gts = targets[1].decompose()[0].cpu().numpy()
            valid_mask = (action_gts != -1)
            for i, pred_action_logit in enumerate(pred_action_logits):
                # pred_action_logit = pred_action_logits[i][~(pred_action_logits[i] == 0).all(dim=1)]
                pred_action_logit = pred_action_logits[i][valid_mask[i]]
                pred_action = pred_action_logit.argmax(dim=-1).cpu().numpy()
                all_action_preds.extend(pred_action)
                action_gt = action_gts[i][~(action_gts[i] == -1)]
                all_action_gts.extend(action_gt)
    if if_confuse:
        utils.plot_confusion_matrix(all_action_gts, all_action_preds, save_path, class_names=action_names)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
