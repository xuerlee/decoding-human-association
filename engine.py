"""
Train and eval functions used in main.py
Reference:
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
import math
import os
import sys
from typing import Iterable

import torch
from scipy.optimize import linear_sum_assignment
import util.misc as utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# collective:
# action_names = ['none', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking']
# activity_names = ['none', 'Crossing', 'Waiting', 'Queuing', 'Walking', 'Talking', 'Empty']
# volleyball:
# action_names = ['blocking', 'digging', 'falling', 'jumping',
#                 'moving', 'setting', 'spiking', 'standing',
#                 'waiting']
# activity_names = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
#                   'l_set', 'l-spike', 'l-pass', 'l_winpoint']
# cafe:
# activity_names = ['Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie']
# jrdb:
action_names = ['standing', 'walking', 'sitting', 'holding sth', 'listening to someone',
                'talking to someone', 'looking at robot', 'looking into sth', 'cycling',
                'looking at sth', 'going upstairs', 'bending', 'typing', 'interaction with door',
                'eating sth', 'talking on the phone', 'going downstairs', 'scootering',
                'pointing at sth', 'pushing', 'reading', 'skating', 'running', 'greeting gestures',
                'writing', 'lying', 'pulling', 'none']
activity_names = ['standing', 'walking', 'sitting', 'holding sth', 'listening to someone',
                  'talking to someone', 'looking at robot', 'looking into sth', 'cycling',
                  'looking at sth', 'going upstairs', 'bending', 'typing', 'interaction with door',
                  'eating sth', 'talking on the phone', 'going downstairs', 'scootering',
                  'pointing at sth', 'pushing', 'reading', 'skating', 'running', 'greeting gestures',
                  'writing', 'lying', 'pulling', 'none']


def matcher_eval(pred_group, oh):
    # oh: n_persons, n_groups
    # pred_group: n_persons, n_queries
    n_group = oh.shape[1]
    n_queries = pred_group.shape[1]
    tgt = oh.T  # num_groups, n_persons
    out = pred_group.T  # num_queries, n_persons
    cost = torch.zeros(n_queries, n_group, device=out.device)
    for i, out_query in enumerate(out):  # n_persons (can be regarded as cls) for certain query: multi cls classification for group
        for j, tgt_group in enumerate(tgt):  # n_persons (can be regarded as cls) for certain group
            inter = (out_query.bool() & tgt_group.bool()).sum()
            union = (out_query.bool() | tgt_group.bool()).sum()
            iou = inter / (union + 1e-6)
            cost[i][j] = 1 - iou
    cost = cost.cpu().numpy()
    indices = linear_sum_assignment(cost)
    return indices  # (out id, tgt id)


def grouping_accuracy(valid_mask, attention_weights, one_hot_gts, one_hot_masks, pred_activity_logits, activity_gts,
                      correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons):
    for i, oh in enumerate(one_hot_gts):
        row_mask = one_hot_masks[i].any(dim=1)  # valid raws, bool tensor
        oh = oh[row_mask]
        mask_valid = one_hot_masks[i][row_mask]
        oh = oh[:, mask_valid[0]]
        aw = attention_weights[i][valid_mask[i]]
        group_ids_person = aw.argmax(dim=-1)
        pred_group = torch.zeros_like(aw)
        for a, b in enumerate(group_ids_person):
            pred_group[a, b] = 1
        pred_activity = pred_activity_logits[i].argmax(dim=-1)

        # for membership accuracy and social accuracy
        out_ids, tgt_ids = matcher_eval(pred_group, oh)
        for out_id, tgt_id in zip(out_ids, tgt_ids):
            correct_person = (oh.T[tgt_id].bool() & pred_group.T[out_id].bool()).sum()
            correct_memberships += correct_person
            if pred_activity[out_id] == activity_gts[i, tgt_id]:
                correct_persons += correct_person
        overall_persons += oh.size(0)

        # for grouping accuracy
        for p, p_group in enumerate(pred_group.T):
            for t, t_group in enumerate(oh.T):
                if torch.equal(p_group, t_group):
                    if pred_activity[p] == activity_gts[i, t]:
                        correct_groups += 1

        # # for mAP
        # for p, p_group in enumerate(pred_group.T):
        #     for t, t_group in enumerate(oh.T):

        overall_groups += oh.size(1)

    return correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons

def train_one_epoch_accum_steps(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, accum_steps: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grp_activity_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('idv_action_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()
    num_iters = len(data_loader)

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

        loss_for_backward = losses / accum_steps
        loss_for_backward.backward()

        should_step = ((iteration + 1) % accum_steps == 0) or ((iteration + 1) == num_iters)
        if should_step:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(grp_activity_class_error=loss_dict_reduced['grp_activity_class_error'])
        # metric_logger.update(grp_activity_class_error=0)
        metric_logger.update(idv_action_class_error=loss_dict_reduced['idv_action_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None:
            global_step = epoch * len(data_loader) + iteration
            writer.add_scalar('Loss/total', loss_value, global_step)

            for k, v in loss_dict_reduced_scaled.items():
                if k.startswith('idv_action_class_error_') or k.startswith('grp_activity_class_error_'):
                    writer.add_scalar(f'Loss_scaled/{k}', v, global_step)
                else:
                    writer.add_scalar(f'Loss_scaled/{k}', v.item(), global_step)

            for k, v in loss_dict_reduced_unscaled.items():
                if k.startswith('idv_action_class_error_') or k.startswith('grp_activity_class_error_'):
                    writer.add_scalar(f'Loss_unscaled/{k}', v, global_step)
                else:
                    writer.add_scalar(f'Loss_unscaled/{k}', v.item(), global_step)

            writer.add_scalar('Error/grp_activity_class_error', loss_dict_reduced['grp_activity_class_error'], global_step)
            writer.add_scalar('Error/idv_action_class_error', loss_dict_reduced['idv_action_class_error'], global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]["lr"], global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grp_activity_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('idv_action_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
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
        metric_logger.update(grp_activity_class_error=loss_dict_reduced['grp_activity_class_error'])
        # metric_logger.update(grp_activity_class_error=0)
        metric_logger.update(idv_action_class_error=loss_dict_reduced['idv_action_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None:
            global_step = epoch * len(data_loader) + iteration
            writer.add_scalar('Loss/total', loss_value, global_step)

            for k, v in loss_dict_reduced_scaled.items():
                if k.startswith('idv_action_class_error_') or k.startswith('grp_activity_class_error_'):
                    writer.add_scalar(f'Loss_scaled/{k}', v, global_step)
                else:
                    writer.add_scalar(f'Loss_scaled/{k}', v.item(), global_step)

            for k, v in loss_dict_reduced_unscaled.items():
                if k.startswith('idv_action_class_error_') or k.startswith('grp_activity_class_error_'):
                    writer.add_scalar(f'Loss_unscaled/{k}', v, global_step)
                else:
                    writer.add_scalar(f'Loss_unscaled/{k}', v.item(), global_step)

            writer.add_scalar('Error/grp_activity_class_error', loss_dict_reduced['grp_activity_class_error'], global_step)
            writer.add_scalar('Error/idv_action_class_error', loss_dict_reduced['idv_action_class_error'], global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]["lr"], global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, dataset, model, criterion, data_loader, device, save_path, if_confuse=False):
    model.eval()
    criterion.eval()

    all_action_preds = []
    all_action_gts = []
    correct_groups = 0
    overall_groups = 0
    correct_persons = 0
    correct_memberships = 0
    overall_persons = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('grp_activity_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('idv_action_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
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
        metric_logger.update(grp_activity_class_error=loss_dict_reduced['grp_activity_class_error'])
        metric_logger.update(idv_action_class_error=loss_dict_reduced['idv_action_class_error'])
        for k, v in loss_dict_reduced.items():
            if k.startswith('idv_action_class_error_') or k.startswith('grp_activity_class_error_') and v is not None:
                metric_logger.update(**{k: v})

        # for final evaluation
        if if_confuse:
            pred_action_logits = outputs['pred_action_logits']
            action_gts = targets[1].decompose()[0].cpu().numpy()
            valid_mask = (action_gts != -1)
            for i, pred_action_logit in enumerate(pred_action_logits):
                pred_action_logit = pred_action_logits[i][valid_mask[i]]
                pred_action = pred_action_logit.argmax(dim=-1).cpu().numpy()
                all_action_preds.extend(pred_action)
                action_gt = action_gts[i][~(action_gts[i] == -1)]
                all_action_gts.extend(action_gt)

            if dataset == 'collective':
                attention_weights = outputs['attention_weights']
                one_hot_gts = targets[3].decompose()[0]
                one_hot_masks = ~targets[3].decompose()[1]
                pred_activity_logits = outputs['pred_activity_logits']
                activity_gts = targets[2].decompose()[0]
                correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons = \
                    grouping_accuracy(valid_mask, attention_weights, one_hot_gts, one_hot_masks, pred_activity_logits, activity_gts,
                                      correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons)

    # final evaluation
    if if_confuse:
        overall_idv_action_acc = (torch.as_tensor(all_action_preds) == torch.as_tensor(all_action_gts)).float().mean()
        overall_idv_action_error = 100 - overall_idv_action_acc * 100
        print('overall_idv_action_error: ', overall_idv_action_error)

        if dataset == 'collective':
            membership_acc = 100 * (correct_memberships / overall_persons)
            social_acc = 100 * (correct_persons / overall_persons)
            grouping_acc = 100 * (correct_groups / overall_groups)
            print('membership accuracy: ', membership_acc)
            print('social accuracy: ', social_acc)
            print('grouping accuracy: ', grouping_acc)

        # confusion matrix
        utils.plot_confusion_matrix(all_action_gts, all_action_preds, save_path, class_names=action_names)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
