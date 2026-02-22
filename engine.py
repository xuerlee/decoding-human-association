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

from collections import Counter, defaultdict
import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment
import util.misc as utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from evaluation.cafe_eval import group_mAP_eval, outlier_metric_from_onehot, outlier_metric, calculateAveragePrecision

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
action_names = ['Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie', 'Individual']
activity_names = ['Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie', 'Individual']
# jrdb:
# action_names = ['standing', 'walking', 'sitting', 'holding sth', 'listening to someone',
#                 'talking to someone', 'looking at robot', 'looking into sth', 'cycling',
#                 'looking at sth', 'going upstairs', 'bending', 'typing', 'interaction with door',
#                 'eating sth', 'talking on the phone', 'going downstairs', 'scootering',
#                 'pointing at sth', 'pushing', 'reading', 'skating', 'running', 'greeting gestures',
#                 'writing', 'lying', 'pulling', 'none']
# activity_names = ['standing', 'walking', 'sitting', 'holding sth', 'listening to someone',
#                   'talking to someone', 'looking at robot', 'looking into sth', 'cycling',
#                   'looking at sth', 'going upstairs', 'bending', 'typing', 'interaction with door',
#                   'eating sth', 'talking on the phone', 'going downstairs', 'scootering',
#                   'pointing at sth', 'pushing', 'reading', 'skating', 'running', 'greeting gestures',
#                   'writing', 'lying', 'pulling', 'none']
# jrdb_simplified:
# action_names = ['walking', 'standing', 'sitting', 'cycling', 'going upstairs', 'bending', 'going downstairs', 'skating', 'scootering', 'running', 'lying']
# activity_names = ['walking', 'standing', 'sitting', 'cycling', 'going upstairs', 'bending', 'going downstairs', 'skating', 'scootering', 'running', 'lying']


# -------------------- CAD ---------------------
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
                      correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons, all_activity_preds, all_activity_gts):
    num_activity_class = pred_activity_logits.shape[-1] - 1
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

        gt_activity_cal = torch.full(pred_activity.shape, num_activity_class, device=activity_gts.device)
        gt_activity_cal[out_ids] = activity_gts[i][tgt_ids]

        all_activity_preds.extend(pred_activity)
        all_activity_gts.extend(gt_activity_cal)


    return correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons, all_activity_preds, all_activity_gts


# ----------------------- JRDB --------------------
def bucket_from_size(sz):
    if sz <= 1: return "G1"
    if sz == 2: return "G2"
    if sz == 3: return "G3"
    if sz == 4: return "G4"
    return "G5+"


def collect_grouping_ap_records_gtboxes(valid_mask, attention_weights, one_hot_gts, one_hot_masks):
    """
    Return:
      records: list of dict(score, tp, bucket)
      npos_bucket: Counter with GT positives per bucket
    """
    records = []
    npos_bucket = Counter()

    for i, oh in enumerate(one_hot_gts):
        row_mask = one_hot_masks[i].any(dim=1)  # valid raws, bool tensor
        oh = oh[row_mask]
        mask_valid = one_hot_masks[i][row_mask]
        oh = oh[:, mask_valid[0]]
        aw = attention_weights[i][valid_mask[i]]

        # pred group per person + its confidence
        pred_gid = aw.argmax(dim=-1)
        score = aw.max(dim=-1).values

        # build pred_group one-hot for matcher_eval
        pred_group = torch.zeros_like(aw)
        pred_group[torch.arange(aw.size(0), device=aw.device), pred_gid] = 1

        # -------- Hungarian assignment between pred queries and GT groups --------
        out_ids, tgt_ids = matcher_eval(pred_group, oh)
        # mapping: pred_query -> gt_group
        map_pred2gt = {int(o): int(t) for o, t in zip(out_ids, tgt_ids)}

        # -------- GT group sizes for bucketing (by GT groups) --------
        # gt group id per person = argmax over columns (because one-hot membership)
        gt_gid = oh.argmax(dim=1).cpu().tolist()
        gt_cnt = Counter(gt_gid)

        # count positives per bucket (GT persons)
        for g in gt_gid:
            npos_bucket[bucket_from_size(gt_cnt[g])] += 1
        npos_bucket["overall"] += len(gt_gid)

        # -------- final TP/FP per person --------
        # person p is TP if mapped(pred_gid[p]) == gt_gid[p]
        for p in range(len(gt_gid)):
            pg = int(pred_gid[p].item())
            mapped = map_pred2gt.get(pg, None)
            tp = 1 if (mapped is not None and mapped == gt_gid[p]) else 0
            bucket = bucket_from_size(gt_cnt[gt_gid[p]])
            records.append({"score": float(score[p].item()), "tp": tp, "bucket": bucket})
            records.append({"score": float(score[p].item()), "tp": tp, "bucket": "overall"})

    return records, npos_bucket


def ap_from_records(records, npos):
    if npos == 0:
        return np.nan
    if len(records) == 0:
        return 0.0

    records = sorted(records, key=lambda x: -x["score"])
    tp = np.array([r["tp"] for r in records], dtype=np.float32)
    fp = 1.0 - tp

    acc_tp = np.cumsum(tp)
    acc_fp = np.cumsum(fp)

    rec = acc_tp / npos
    prec = acc_tp / (acc_tp + acc_fp + 1e-8)

    ap, _, _, _ = calculateAveragePrecision(rec.tolist(), prec.tolist())
    return ap * 100.0


# -------------- cafe -----------------
def build_groups_dicts_from_tensors(args, meta, valid_mask, attention_weights, one_hot_gts, one_hot_masks,
                                    pred_activity_logits, activity_gts, activity_masks):
    gt_groups_ids = defaultdict(list)
    gt_groups_activity = defaultdict(list)

    pred_groups_ids = defaultdict(list)
    pred_groups_activity = defaultdict(list)
    pred_groups_scores = defaultdict(list)

    B = pred_activity_logits.shape[0]
    prob = F.softmax(pred_activity_logits, dim=-1)
    pred_act = prob.argmax(dim=-1)
    pred_score = prob.max(dim=-1).values

    for i in range(B):
        if args.dataset == 'jrdb' or args.dataset == 'jrdb_group':
            sid = meta[i]["sid"]
            src_fid = meta[i]["src_fid"]
            clip_key = f"{sid}, {src_fid}"
        elif args.dataset == 'cafe':
            sid = meta[i]["sid"]
            cid = meta[i]["cid"]
            clip_key = f"{sid},{cid}"

        row_mask = one_hot_masks[i].any(dim=1)  # valid raws, bool tensor
        oh = one_hot_gts[i][row_mask]
        mask_valid = one_hot_masks[i][row_mask]
        oh = oh[:, mask_valid[0]]

        gt_act = activity_gts[i][activity_masks[i]]

        gt_gid_dict = defaultdict(list)
        gt_act_dict = defaultdict(set)
        for g in range(oh.shape[1]):
            members = torch.where(oh[:, g] == 1)[0].tolist()
            gt_gid_dict[g].extend(members)                  # group_id = g (column index)
            gt_act_dict[g].add(int(gt_act[g]))
        gt_groups_ids[clip_key].append(gt_gid_dict)
        gt_groups_activity[clip_key].append(gt_act_dict)

        # ---------- Pred membership ----------
        aw = attention_weights[i][valid_mask[i]]
        gids = aw.argmax(dim=-1)

        pred_gid_dict = defaultdict(list)
        pred_act_dict = defaultdict(set)
        pred_score_dict = defaultdict(set)

        # build members list for each predicted group (query id)
        for p in range(aw.shape[0]):
            q = int(gids[p].item())  # group id
            pred_gid_dict[q].append(p)  # person id is appended

        # attach act + score for each predicted group id
        for q in pred_gid_dict.keys():
            pred_act_dict[q].add(int(pred_act[i, q].item()))
            pred_score_dict[q].add(float(pred_score[i, q].item()))

        pred_groups_ids[clip_key].append(pred_gid_dict)
        pred_groups_activity[clip_key].append(pred_act_dict)
        pred_groups_scores[clip_key].append(pred_score_dict)

    return gt_groups_ids, gt_groups_activity, pred_groups_ids, pred_groups_activity, pred_groups_scores


def group_overlap_maxnorm(det_members, gt_members):
    """ |inter| / max(|det|, |gt|) """
    det_set = set(det_members)
    gt_set = set(gt_members)
    inter = len(det_set & gt_set)
    denom = max(len(det_set), len(gt_set))
    return 0.0 if denom == 0 else inter / denom


def group_prf_eval(gt_groups_ids, pred_groups_ids, thresh=0.5, min_group_size=2, ignore_pred_gid_minus1=True):
    """
    Evaluate group detection P/R/F1 following:
      TP if max_gt |inter| / max(|det|,|gt|) > thresh, with 1-1 matching.

    Args:
      gt_groups_ids: dict clip_key -> [ {gt_gid: [person_ids]} ]
      pred_groups_ids: dict clip_key -> [ {pred_gid: [person_ids]} ]
      thresh: overlap threshold (paper uses 0.5)
      min_group_size: only evaluate groups with size>=min_group_size (typical: 2)
      ignore_pred_gid_minus1: if your pred dict uses gid=-1 for "no group", skip it

    Returns:
      precision, recall, f1, (TP, FP, FN)
    """
    TP = 0
    FP = 0
    FN = 0

    for clip_key in pred_groups_ids.keys():
        if clip_key not in gt_groups_ids:
            continue

        gt_dict = gt_groups_ids[clip_key][0]     # {gid: [pids]}
        pred_dict = pred_groups_ids[clip_key][0] # {gid: [pids]}

        # filter GT groups (usually size>=2)
        gt_groups = []
        for gid, members in gt_dict.items():
            if len(members) >= min_group_size:
                gt_groups.append((gid, members))

        # filter Pred groups
        pred_groups = []
        for gid, members in pred_dict.items():
            if ignore_pred_gid_minus1 and gid == -1:
                continue
            if len(members) >= min_group_size:
                pred_groups.append((gid, members))

        matched_gt = set()  # store matched GT gid

        # greedy matching: for each pred group, match best available GT group
        for pgid, pmembers in pred_groups:
            best_score = 0.0
            best_gt = None
            for ggid, gmembers in gt_groups:
                if ggid in matched_gt:
                    continue
                score = group_overlap_maxnorm(pmembers, gmembers)
                if score > best_score:
                    best_score = score
                    best_gt = ggid

            if best_gt is not None and best_score > thresh:
                TP += 1
                matched_gt.add(best_gt)
            else:
                FP += 1

        # any unmatched GT groups are FN
        FN += (len(gt_groups) - len(matched_gt))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))

    return precision * 100.0, recall * 100.0, f1 * 100.0, (TP, FP, FN)


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

            # writer.add_scalar('Error/grp_activity_class_error', loss_dict_reduced['grp_activity_class_error'], global_step)
            # writer.add_scalar('Error/idv_action_class_error', loss_dict_reduced['idv_action_class_error'], global_step)
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

            # writer.add_scalar('Error/grp_activity_class_error', loss_dict_reduced['grp_activity_class_error'], global_step)
            # writer.add_scalar('Error/idv_action_class_error', loss_dict_reduced['idv_action_class_error'], global_step)
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

    all_activity_preds = []
    all_activity_gts = []

    all_oh = []
    all_aw = []

    correct_groups = 0
    overall_groups = 0
    correct_persons = 0
    correct_memberships = 0
    overall_persons = 0

    if dataset == 'jrdb' or dataset == 'jrdb_group':
        all_records = []
        npos_bucket = Counter()
        gt_groups_ids_all = defaultdict(list)
        gt_groups_activity_all = defaultdict(list)
        pred_groups_ids_all = defaultdict(list)
        pred_groups_activity_all = defaultdict(list)
        pred_groups_scores_all = defaultdict(list)
    elif dataset == 'cafe':
        gt_groups_ids_all = defaultdict(list)
        gt_groups_activity_all = defaultdict(list)
        pred_groups_ids_all = defaultdict(list)
        pred_groups_activity_all = defaultdict(list)
        pred_groups_scores_all = defaultdict(list)

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
                correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons, all_activity_preds, all_activity_gts = \
                    grouping_accuracy(valid_mask, attention_weights, one_hot_gts, one_hot_masks, pred_activity_logits, activity_gts,
                                      correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons,
                                      all_activity_preds, all_activity_gts)

            if dataset == 'jrdb' or dataset == 'jrdb_group':
                attention_weights = outputs['attention_weights']
                one_hot_gts = targets[3].decompose()[0]
                one_hot_masks = ~targets[3].decompose()[1]
                pred_activity_logits = outputs['pred_activity_logits']
                activity_gts = targets[2].decompose()[0]
                activity_masks = ~targets[2].decompose()[1]

                # for group activity error
                correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons, all_activity_preds, all_activity_gts = \
                    grouping_accuracy(valid_mask, attention_weights, one_hot_gts, one_hot_masks, pred_activity_logits, activity_gts,
                                      correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons,
                                      all_activity_preds, all_activity_gts)

                # for G1 AP, G2 AP, G3 AP, G4 AP, G5+ AP, Overall AP
                records_b, npos_b = collect_grouping_ap_records_gtboxes(valid_mask, attention_weights, one_hot_gts,
                                                                        one_hot_masks)
                all_records.extend(records_b)
                npos_bucket.update(npos_b)

                # for p, r, f1
                (gt_groups_ids_b, gt_groups_activity_b,
                 pred_groups_ids_b, pred_groups_activity_b, pred_groups_scores_b) = build_groups_dicts_from_tensors(
                    args, meta, valid_mask,
                    attention_weights, one_hot_gts, one_hot_masks,
                    pred_activity_logits, activity_gts, activity_masks
                )
                for ck in gt_groups_ids_b.keys():
                    gt_groups_ids_all[ck] = gt_groups_ids_b[ck]
                    gt_groups_activity_all[ck] = gt_groups_activity_b[ck]

                for ck in pred_groups_ids_b.keys():
                    pred_groups_ids_all[ck] = pred_groups_ids_b[ck]
                    pred_groups_activity_all[ck] = pred_groups_activity_b[ck]
                    pred_groups_scores_all[ck] = pred_groups_scores_b[ck]

            if dataset == 'cafe':
                attention_weights = outputs['attention_weights']
                one_hot_gts = targets[3].decompose()[0]
                one_hot_masks = ~targets[3].decompose()[1]
                pred_activity_logits = outputs['pred_activity_logits']
                activity_gts = targets[2].decompose()[0]
                activity_masks = ~targets[2].decompose()[1]

                # for group activity error
                correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons, all_activity_preds, all_activity_gts = \
                    grouping_accuracy(valid_mask, attention_weights, one_hot_gts, one_hot_masks, pred_activity_logits, activity_gts,
                                      correct_groups, overall_groups, correct_persons, correct_memberships, overall_persons,
                                      all_activity_preds, all_activity_gts)

                # for Group mAP
                (gt_groups_ids_b, gt_groups_activity_b,
                 pred_groups_ids_b, pred_groups_activity_b, pred_groups_scores_b) = build_groups_dicts_from_tensors(
                    args, meta, valid_mask,
                    attention_weights, one_hot_gts, one_hot_masks,
                    pred_activity_logits, activity_gts, activity_masks
                )

                B = attention_weights.shape[0]
                for i in range(B):
                    row_mask = one_hot_masks[i].any(dim=1)  # valid raws, bool tensor
                    oh = one_hot_gts[i][row_mask]
                    mask_valid = one_hot_masks[i][row_mask]
                    oh = oh[:, mask_valid[0]]
                    aw = attention_weights[i][valid_mask[i]]
                    all_oh.append(oh.detach().cpu())
                    all_aw.append(aw.detach().cpu())

                for ck in gt_groups_ids_b.keys():
                    gt_groups_ids_all[ck] = gt_groups_ids_b[ck]
                    gt_groups_activity_all[ck] = gt_groups_activity_b[ck]

                for ck in pred_groups_ids_b.keys():
                    pred_groups_ids_all[ck] = pred_groups_ids_b[ck]
                    pred_groups_activity_all[ck] = pred_groups_activity_b[ck]
                    pred_groups_scores_all[ck] = pred_groups_scores_b[ck]


    # final evaluation
    if if_confuse:
        overall_idv_action_acc = (torch.as_tensor(all_action_preds) == torch.as_tensor(all_action_gts)).float().mean()
        overall_idv_action_error = 100 - overall_idv_action_acc * 100
        print('overall_idv_action_error: ', overall_idv_action_error)

        overall_grp_activity_acc = (torch.as_tensor(all_activity_preds) == torch.as_tensor(all_activity_gts)).float().mean()
        overall_grp_activity_error = 100 - overall_grp_activity_acc * 100
        print('overall_grp_activity_error: ', overall_grp_activity_error)

        if dataset == 'collective':
            membership_acc = 100 * (correct_memberships / overall_persons)
            social_acc = 100 * (correct_persons / overall_persons)
            grouping_acc = 100 * (correct_groups / overall_groups)
            print('CAD membership accuracy: ', membership_acc)
            print('CAD social accuracy: ', social_acc)
            print('CAD grouping accuracy: ', grouping_acc)

        elif dataset == 'jrdb' or dataset == 'jrdb_group':
            for b in ["G1", "G2", "G3", "G4", "G5+", "overall"]:
                recs = [r for r in all_records if r["bucket"] == b]
                ap = ap_from_records(recs, npos_bucket[b])
                print(b, "AP:", ap)

            p, r, f1, (TP, FP, FN) = group_prf_eval(
                gt_groups_ids_all, pred_groups_ids_all,
                thresh=0.5, min_group_size=2
            )
            print("group_P@0.5:", p)
            print("group_R@0.5:", r)
            print("group_F1@0.5:", f1)

        elif dataset == 'cafe':
            categories = [{"id": i, "name": n} for i, n in enumerate(activity_names)]
            mAP10, APs10 = group_mAP_eval(gt_groups_ids_all, gt_groups_activity_all,
                                          pred_groups_ids_all, pred_groups_activity_all, pred_groups_scores_all,
                                          categories, thresh=1.0)
            mAP05, APs05 = group_mAP_eval(gt_groups_ids_all, gt_groups_activity_all,
                                          pred_groups_ids_all, pred_groups_activity_all, pred_groups_scores_all,
                                          categories, thresh=0.5)
            outlier_from_onehot = outlier_metric_from_onehot(all_oh, all_aw)
            num_class = len(activity_names)
            outlier = outlier_metric(gt_groups_ids_all, gt_groups_activity_all, pred_groups_ids_all, pred_groups_activity_all, num_class-1)
            print("CAFE group_mAP@1.0:", mAP10)
            print("CAFE group_mAP@0.5:", mAP05)
            print("CAFE outlier_mIoU:", outlier)
            print("CAFE outlier_mIoU_from_onehot:", outlier_from_onehot)

            p, r, f1, (TP, FP, FN) = group_prf_eval(
                gt_groups_ids_all, pred_groups_ids_all,
                thresh=0.5, min_group_size=2
            )

            print("group_P@0.5:", p)
            print("group_R@0.5:", r)
            print("group_F1@0.5:", f1)


        # confusion matrix
        utils.plot_confusion_matrix(all_action_gts, all_action_preds, save_path, class_names=action_names)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
