# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from util.misc import crop_to_original


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the one-hot matrices (target) and cross attention weights showing the individual grouping predictions

    For efficiency reasons, the targets don't include the empty group. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1 row-to-1 row matching of the best predictions,
    while the others are un-matched (and thus treated as empty groups).
    """

    def __init__(self, cost_activity_class: float = 1, cost_action_class: float = 1, cost_bce: float = 1, cost_size: float = 1):
        """Creates the matcher

        Params:
            cost_activity_class: This is the relative weight of the group activity classification error in the matching cost
            cost_action_class: This is the relative weight of the individual action classes consistency in the matching cost
            cost_bce: This is the relative weight of the BCE error of one-hot grouping matrices and cross attention weights in the matching cost
        """
        super().__init__()
        self.cost_activity_class = cost_activity_class
        self.cost_action_class = cost_action_class
        self.cost_bce = cost_bce
        self.cost_size = cost_size
        assert cost_activity_class != 0 or cost_action_class != 0 or cost_bce != 0 or cost_size != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching, only calculate valid areas of targets

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_activity_logits": Tensor of dim [batch_size, num_queries, num_activity_classes] with the group activity
                  classification logits
                 "attention_weights": Tensor of dim [batch_size, num_persons_max, num_queries] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 activity classes: Tensor of dim [num_target_groups] (where num_target_groups is the number of ground-truth
                           groups in the target) containing the group activity class labels
                 one-hot matrices: Tensor of dim [num_persons, num_target_groups] indicating the target grouping arrangement

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_groups)
        """

        bs, num_queries, num_activity_classes = outputs["pred_activity_logits"].shape  # B, num_queries, num_activity_classes

        # We flatten to compute the cost matrices in a batch
        # approximate negative log likelihood cost, add negative (omit log for cost) later: cost_class = -out_prob[:, tgt_ids]
        # F.cross entropy = softmax + negative log likelihood loss; F.binary_cross_entropy_with_logits = sigmoid + BCE
        # out_activity_prob = outputs["pred_activity_logits"].flatten(0, 1).softmax(-1).view(bs, num_queries, -1)  # get likelihood, [batch_size, num_queries, num_activity_classes]
        out_activity_prob = outputs["pred_activity_logits"].view(bs, num_queries, -1)  # [batch_size, num_queries, num_activity_classes]
        out_attw = outputs['attention_weights']  # B, n_max, num_queries

        # target labels and assignments
        tgt_activity_ids, mask_ids = targets[0].decompose()  # B, num_group_max
        tgt_one_hot, mask_one_hot = targets[1].decompose()  # B, n_max, num_groups_max

        valid_areas_ids = crop_to_original(mask_ids)  # batch size, 2 (min, max)
        valid_areas_one_hot = crop_to_original(mask_one_hot)  # batch size, 4 (ymin ymax xmin xmax)

        indices = []
        for b in range(bs):
            tgt_activity_ids_b = tgt_activity_ids[b][valid_areas_ids[b][0]: valid_areas_ids[b][1]]  # [num_group]
            tgt_activity_ids_b = F.one_hot(tgt_activity_ids_b, num_activity_classes)  # num_group, num_activity_cls
            tgt_one_hot_b = tgt_one_hot[b][valid_areas_one_hot[b][0]: valid_areas_one_hot[b][1], valid_areas_one_hot[b][2]: valid_areas_one_hot[b][3]]  # n_persons, n_groups
            n_group = len(tgt_activity_ids_b)
            n_person = tgt_one_hot_b.shape[0]
            out_activity_prob_b = out_activity_prob[b]  # [num_queries, num_classes]
            out_attw_b = out_attw[b][0: n_person, :]  # [num_persons, num_queries]
            # out_attw_b = out_attw_b.softmax(dim=-1)

            tgt_one_hot_b = tgt_one_hot_b.T  # num_groups, n_persons
            out_attw_b = out_attw_b.T  # num_queries, n_persons

            # tgt_size = tgt_one_hot_b.sum(dim=-1)
            # out_size = out_attw_b.sum(dim=-1)

            grouping_cost = torch.zeros(num_queries, n_group, device=out_attw.device)
            activity_cost = torch.zeros(num_queries, n_group, device=out_attw.device)
            # size_cost = torch.zeros(num_queries, n_group, device=out_attw.device)
            for i, out_attw_b_query in enumerate(out_attw_b):  # n_persons (can be regarded as cls) for certain query: multi cls classification for group
                for j, tgt_one_hot_b_group in enumerate(tgt_one_hot_b):  # n_persons (can be regarded as cls) for certain group
                    grouping_cost[i][j] = F.binary_cross_entropy_with_logits(out_attw_b_query.float(), tgt_one_hot_b_group.float())  # direction: -> smaller cost  1.  multi cls(persons) classification for groups
                    # activity_cost[i][j] = F.cross_entropy(out_activity_prob_b[i].float(), tgt_activity_ids_b[j])
                    # activity_cost[i][j] = -out_activity_prob_b[i].float() * tgt_activity_ids_b[j] - (1 - out_activity_prob_b[i].float()) * (1 - tgt_activity_ids_b[j])  # direction: -> smaller cost  0.
                    activity_cost[i][j] = F.binary_cross_entropy_with_logits(out_activity_prob_b[i].float(), tgt_activity_ids_b[j].float())


            cost_b = self.cost_bce * grouping_cost + self.cost_activity_class * activity_cost
            cost_b = cost_b.cpu().numpy()
            indices_b = linear_sum_assignment(cost_b)
            indices.append(indices_b)  # indices: B, 2 (prediction_id, target_id), num_groups
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  # list with bs length, each element: (src id, tgt id)


def build_matcher(args):
    # TODO: ADD cost_action
    return HungarianMatcher(cost_activity_class=args.set_cost_activity_class, cost_action_class=args.set_cost_action_class, cost_bce=args.set_cost_bce, cost_size=args.set_cost_size)
