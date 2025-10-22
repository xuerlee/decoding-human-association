# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

from util.misc import (NestedTensor, nested_tensor_from_tensor_list, nested_tensor_from_fm_list,
                       crop_to_original, binary_label_smoothing, accuracy, per_class_accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone_i3d import build_backboneI3D
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
import time

class DETR(nn.Module):
    """ This is the DETR module that performs group recognition and actions classification"""
    def __init__(self, backbone, transformer, feature_channels, num_action_classes, num_activity_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: RoI Align features for individuals for each frame in one batch. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            feature_channels: number of feature channels output by the feature extraction part
            num_aciton_classes: number of individual action categories
            num_activity_classes: number of group activity categories
            num_queries: number of queries for decoder, ie group detection slot. This is the maximal number of groups
                         DETR can detect in a single image.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        # self.hidden_dim = 256*7*7
        self.action_class_embed = nn.Linear(self.hidden_dim, num_action_classes)
        self.activity_class_embed = nn.Linear(self.hidden_dim, num_activity_classes + 1)  # including empty groups
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.aw_embed = MLP(num_queries, self.hidden_dim, num_queries, 2)
        self.dropout = nn.Dropout(p=0.1)  # set zeros randomly, no influences on valid mask
        self.backbone = backbone
        self.aux_loss = aux_loss


    def forward(self, features: NestedTensor, bboxes: NestedTensor, meta):
        """samples are NestedTensor of the stacked feature maps/images before roi align, including feature maps and masks

            It returns a dict with the following elements:
               - "pred_activity_logits (decoder)": the classification logits for activities including no groups for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_action_logits (encoder)": the classification logits for actions for all input roi aligned persons.
                                Shape= [batch_size x num_persons x (num_classes + 1)]
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(features, (list, torch.Tensor)):
            features = nested_tensor_from_fm_list(features)

        src_f, mask_f = features.decompose()  # NestedTensor features/images  B, T, C, H, W
        # valid_areas_f = crop_to_original(mask_f)  # batch size, 4 (ymin ymax xmin xmax)

        src_b, mask_b = bboxes.decompose()  # B, n_max, 4
        valid_areas_b = crop_to_original(mask_b)  # batch size, 4 (ymin ymax xmin xmax)
        boxes_features, pos, mask = self.backbone(src_f, src_b, valid_areas_b, meta)  # roi align + position encoding  mask: B, n_max
        hs, memory, attention_weights = self.transformer(boxes_features, mask, self.query_embed.weight, pos)  # hs: num_dec_layers, B*T, num_queries, hidden_dim; memory: B*T, n_max, hidden_dim; AW: B*T, num_queries, n_max
        # hs, memory, attention_weights = self.transformer(boxes_features, mask, self.query_embed.weight, None)  # without positional embeddings

        # without Transformer (for debug)
        # B = src_f.shape[0]
        # n_max = src_b.shape[1]
        # boxes_features = self.dropout(boxes_features)
        # mask = ~mask.view(B, n_max)  # B, n_max
        # outputs_action_class = self.action_class_embed(boxes_features)  # B, n_max, num_action_classes
        # outputs_action_class = self.dropout(outputs_action_class)
        # outputs_action_class = outputs_action_class * mask.unsqueeze(-1)
        # action_scores = outputs_action_class
        # out = {'pred_action_logits': action_scores}

        # individual action classfication
        B = src_f.shape[0]
        n_max = src_b.shape[1]
        memory = memory.view(B, n_max, self.hidden_dim)
        memory = self.dropout(memory)
        mask = ~mask.view(B, n_max)
        outputs_action_class = self.action_class_embed(memory)  # B, n_max, num_action_classes
        outputs_action_class = self.dropout(outputs_action_class)
        # outputs_action_class = outputs_action_class * mask.unsqueeze(-1)
        action_scores = outputs_action_class

        # group activity classification
        hs = hs.view(-1, B, self.num_queries, self.hidden_dim)
        outputs_activity_class = self.activity_class_embed(hs)  # num_dec_layers, B, num_queries, num_activity_classes
        activity_scores = outputs_activity_class

        # for grouping based on attention weights
        attention_weights = attention_weights.transpose(1, 2).contiguous()  # B, n_max, num_queries
        attention_weights = self.aw_embed(attention_weights)  # B, n_max, num_queries
        # attention_weights = F.softmax(attention_weights, dim=2)  # make the sum of logits as 1  (each person belongs to which group)
        # attention_weights = attention_weights * mask.unsqueeze(-1)  # B, n_max, num_queries

        out = {'pred_action_logits': action_scores, 'pred_activity_logits': activity_scores[-1], 'attention_weights': attention_weights}  # activity scores: only take the output of the last later here

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(action_scores, activity_scores, attention_weights)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, action_scores, activity_scores, attention_weights):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_action_logits': action_scores, 'pred_activity_logits': a, 'attention_weights': attention_weights} for a in activity_scores[:-1]]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth groups and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and grouping)
    """
    # TODO: add action class consistency: the relations between individual actions and group activities
    def __init__(self, dataset, num_action_classes, num_activity_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_action_classes: number of individual actions categories
            num_activity_classes: number of group activities categories
            matcher: module able to compute a matching between targets and predictions
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the empty groups categories
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.dataset = dataset
        self.num_action_classes = num_action_classes
        self.num_activity_classes = num_activity_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        # for empty group prediction
        self.eos_coef = eos_coef
        self.empty_weight = torch.ones(self.num_activity_classes + 1)
        self.empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_activity_weight', self.empty_weight)
        self.register_buffer("G2I_mask", self._build_G2I_mask(self.num_activity_classes, self.num_action_classes))

    def loss_activity_labels(self, outputs, targets, indices, num_groups, log=True):
        """group activity classification loss (NLL)
        targets dicts must contain the key "pred_activity_logits"
        """
        assert 'pred_activity_logits' in outputs
        out_activity_logits = outputs['pred_activity_logits']  # [B, num_queries, num_activity_classes]
        tgt_activity_ids, mask_ids = targets[1].decompose()  # B, num_group_max

        '''
        1 get batch idx and src idx from matcher
        2 change orders of targets according to tgt inx from matcher to match the prediction orders 
        3 construct target tensor for loss calculation, with same shape as output logits
        4 fill the target tensor by targets with the same order as corresponding matched logits
        '''
        idx = self._get_src_permutation_idx(indices)  # length: sum of the number of groups in a whole batch

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(tgt_activity_ids, indices)])  # t: tgt_activity_ids_b; J: macthed_tgt_id for each batch (change orders to match the prediction)
        target_classes = torch.full(out_activity_logits.shape[:2], self.num_activity_classes,
                                    dtype=torch.int64, device=out_activity_logits.device)  # the id of the empty group is num_activity_class
        target_classes[idx] = target_classes_o  # target_classes[idx]: set other output matched group idxes except for empty groups  # bs, num_queries

        self.empty_weight = self.empty_weight.to(out_activity_logits.device)
        loss_ce = F.cross_entropy(out_activity_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        losses = {'loss_activity': loss_ce}

        if log:
            losses['activity_class_error'] = 100 - accuracy(out_activity_logits[idx], target_classes_o)[0]  # 100 - accuracy

            class_acc = per_class_accuracy(out_activity_logits[idx], target_classes_o, num_classes=out_activity_logits.shape[-1])
            for i, acc in enumerate(class_acc):
                if not math.isnan(acc):
                    losses[f'activity_class_error_{i}'] = 100 - acc
        return losses

    def loss_action_labels(self, outputs, targets, indices, num_groups, log=True):
        """individual action lassification loss (NLL)
        targets dicts must contain the key "pred_action_logits" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_action_logits' in outputs
        src_logits = outputs['pred_action_logits']  # [B, n_max, num_action_classes]
        tgt_action_ids, mask_ids = targets[0].decompose()  # B, n_max
        idx = torch.where(tgt_action_ids != -1)
        tgt_action_ids = tgt_action_ids[idx]  # n_persons in B
        src_logits = src_logits[idx]  # n_persons in B, num_action_classes  # class is always at dim1
        # loss_ce = F.cross_entropy(src_logits, tgt_action_ids, label_smoothing=0.05)
        loss_ce = F.cross_entropy(src_logits, tgt_action_ids, ignore_index=-1)  # ignore index instead of multiplying mask
        losses = {'loss_action': loss_ce}

        if log:
            losses['action_class_error'] = 100 - accuracy(src_logits, tgt_action_ids)[0]
            class_acc = per_class_accuracy(src_logits, tgt_action_ids, num_classes=src_logits.shape[-1])
            for i, acc in enumerate(class_acc):
                if not math.isnan(acc):
                    losses[f'action_class_error_{i}'] = 100 - acc
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_groups):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty groups
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_activity_logits']
        device = pred_logits.device
        tgt_activity_ids, mask_ids = targets[1].decompose()  # B, num_group_max
        tgt_lengths = torch.tensor([(v != -1).sum() for v in tgt_activity_ids],device=device)

        # Count the number of predictions that are NOT "no-group" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)  # judge if the predicted class is the empty group class
        card_err = F.l1_loss(card_pred.float(), tgt_lengths)
        losses = {'cardinality_error': card_err}
        return losses

    def loss_grouping(self, outputs, targets, indices, num_groups):
        """Compute the losses related to the grouping results, using the binary cross entropy loss
        """
        # TODO: cross entropy
        assert 'attention_weights' in outputs
        src_aw = outputs['attention_weights']  # B, n_max, num_queries

        tgt_one_hot_ini, mask_one_hot = targets[-1].decompose()  # B, n_max, num_groups_max
        tgt_one_hot_ini = tgt_one_hot_ini.transpose(1, 2)  # B, num_groups_max, n_max  # regard persons as cls
        mask_one_hot = mask_one_hot.transpose(1, 2)  # B, num_groups_max, n_max

        idx = self._get_src_permutation_idx(indices)
        target_one_hot_o = torch.cat([t[J] for t, (_, J) in zip(tgt_one_hot_ini, indices)])  # t: tgt_activity_ids_b; J: macthed_tgt_id for each batch (change orders to match the prediction)
        target_one_hot = torch.full(src_aw.shape, 0,
                                    dtype=torch.int, device=src_aw.device)
        target_one_hot = target_one_hot.transpose(1, 2)  # B, num_queries, n_max
        target_one_hot[idx] = target_one_hot_o  # targrt_one_hot[batch_idx, src_idx] = targrt_one_hot_o  # B, num_queries, n_max

        mask_one_hot = ~mask_one_hot
        mask_one_hot_o = torch.cat([t[J] for t, (_, J) in zip(mask_one_hot, indices)])
        valid_mask = torch.full(src_aw.shape, False, dtype=torch.bool, device=src_aw.device)
        valid_mask = valid_mask.transpose(1, 2)   # B, num_queries, n_max
        valid_mask[idx] = mask_one_hot_o

        # loss w.s.t assigning people to group
        pos = target_one_hot[valid_mask].sum()
        neg = target_one_hot[valid_mask].numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(1., 50.)
        pos_weight = pos_weight.to(target_one_hot.device)
        loss_grouping = F.binary_cross_entropy_with_logits(src_aw.transpose(1, 2)[valid_mask], target_one_hot[valid_mask].float(), pos_weight=pos_weight)

        # loss w.s.t grouping people
        # target_group = target_one_hot.transpose(1, 2).argmax(-1)  # B, n_max
        # loss_grouping = F.cross_entropy(src_aw.transpose(1, 2), target_group)

        losses = {}
        # losses['loss_grouping'] = loss_grouping.sum() / num_groups
        losses['loss_grouping'] = loss_grouping

        return losses

    def loss_action_group_consistency(self, outputs, targets, indices, num_groups):
        out_activity_logits = outputs['pred_activity_logits']  # [B, num_queries, num_activity_classes]
        out_action_logits = outputs['pred_action_logits']  # [B, n_max, num_action_classes]
        src_aw = outputs['attention_weights']  # B, n_max, num_queries
        G2I_mask = self._build_G2I_mask(self.num_activity_classes, self.num_action_classes).to(src_aw.device)  # num_group_classes, num_action_classes
        # G2I_mask = F.softmax(G2I_mask.float(), dim=1)  # one-hot matrix, transfer group labels to related action labels
        G2I_mask = binary_label_smoothing(G2I_mask.float(), 0.1, False)  # one-hot matrix, transfer group labels to related action labels

        out_action_probs = F.softmax(out_action_logits, dim=-1)  # B, n_max, num_action_classes
        out_group_labels = out_activity_logits.argmax(dim=-1)  # B, num_queries
        group_expected_actions = G2I_mask[out_group_labels]  # B, num_queries, num_action_classes (one-hot)  select the correspoding rows in g2i mask to get the corresponding expected action distribution

        _, valid_mask = targets[0].decompose()
        valid_person_mask = ~valid_mask  # B, n_max
        valid_group_mask = (out_group_labels != self.num_activity_classes)  # B, num_queries  mask empty groups

        loss = 0  # multi actions - multi actions
        B = valid_mask.shape[0]
        for b in range(B):
            src_aw_b = src_aw[b][valid_person_mask[b]]  # n_b, num_queries
            out_action_probs_b = out_action_probs[b][valid_person_mask[b]]  # n_b, num_action_classes
            actions_in_group_dist_b = torch.matmul(src_aw_b.T, out_action_probs_b)  # [n_queries, num_action_classes]  distribution of actions in group
            group_expected_actions_b = group_expected_actions[b]  # n_queries, num_action_classes

            num_valid = (valid_group_mask[b] != False).sum()
            if num_valid == 0:
                loss_b = torch.tensor(0., device=src_aw.device, requires_grad=True)
            else:
                loss_b = F.binary_cross_entropy(
                    actions_in_group_dist_b[valid_group_mask[b]],
                    group_expected_actions_b[valid_group_mask[b]].float())
            loss += loss_b

        losses = {}
        losses['loss_consistency'] = loss
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])  # tensor with length of sum of num of groups in one batch, indicating batch id: e.g.: tensor([0, 0, 1, 1, 1, 1, 1, 1, 1])
        src_idx = torch.cat([src for (src, _) in indices])  # tensor of src id, e.g.: tensor([4, 8, 1, 2, 3, 4, 5, 6, 7])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _build_G2I_mask(self, G, A):
        mask = torch.zeros(G+1, A)
        if self.dataset == 'collective':
            mask = torch.eye(G+1, A)
        if self.dataset == 'volleyball':
            pass
        return mask

    def get_loss(self, loss, outputs, targets, indices, num_groups, **kwargs):
        loss_map = {
            'activity': self.loss_activity_labels,
            'action': self.loss_action_labels,
            'cardinality': self.loss_cardinality,
            'grouping': self.loss_grouping,
            'consistency': self.loss_action_group_consistency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_groups, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of tensors
                      action classes: Tensor of dim [num_individuals] (where num_individuals is the number of ground-truth
                      individuals in the target) containing the individual action class labels
                      activity classes: Tensor of dim [num_target_groups] (where num_target_groups is the number of ground-truth
                      groups in the target) containing the group activity class labels
                      one-hot matrices: Tensor of dim [num_persons, num_target_groups] indicating the target grouping arrangement
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'pred_action_logits'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets[1:])  # src id, tgt id
        # Compute the average number of target groups accross all nodes, for normalization purposes
        num_groups = (targets[1].decompose()[0] != -1).sum()
        num_groups = torch.as_tensor([num_groups], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_groups)
        num_groups = torch.clamp(num_groups / get_world_size(), min=1).item()  # every gpu has at least on positive sample

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_groups))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         indices = self.matcher(aux_outputs, targets[1:])
        #         for loss in self.losses:
        #             kwargs = {}
        #             if loss == 'avtivity':
        #                 # Logging is enabled only for the last layer
        #                 kwargs = {'log': False}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_groups, **kwargs)
        #             l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # FIXME: num class of volleyball
    num_action_classes = 6 if args.feature_file != 'volleyball' else 9  # not sure
    num_activity_classes = 6 if args.feature_file != 'volleyball' else 4  # not sure

    device = torch.device(args.device)

    if args.input_format == 'feature':
        backbone = build_backbone(args)
    elif args.input_format == 'image':
        backbone = build_backboneI3D(args)
    else:
        raise ValueError(f'import format {args.input_format} not supported, options: image or feature')

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        feature_channels=args.feature_channels,
        num_action_classes=num_action_classes,
        num_activity_classes=num_activity_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_action': args.action_loss_coef, 'loss_activity': args.activity_loss_coef, 'loss_grouping': args.grouping_loss_coef, 'loss_consistency': args.consistency_loss_coef}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # losses = ['activity', 'grouping', 'action', 'cardinality', 'consistency']
    losses = ['activity', 'grouping', 'action', 'cardinality']
    # losses = ['activity', 'grouping', 'action']
    # losses = ['action']
    criterion = SetCriterion(args.feature_file, num_action_classes, num_activity_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    # postprocessors = {'bbox': PostProcess()}

    return model, criterion
