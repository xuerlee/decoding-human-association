"""
Modules to compute the matching cost and solve the corresponding.
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

    def __init__(self, cost_activity_class: float = 1, cost_action_class: float = 1, cost_group: float = 1, cost_size: float = 1):
        """Creates the matcher

        Params:
            cost_activity_class: This is the relative weight of the group activity classification error in the matching cost
            cost_action_class: This is the relative weight of the individual action classes consistency in the matching cost
            cost_group: This is the relative weight of the cost of one-hot grouping matrices and cross attention weights in the matching cost
        """
        super().__init__()
        self.cost_activity_class = cost_activity_class
        self.cost_action_class = cost_action_class
        self.cost_group = cost_group
        self.cost_size = cost_size
        assert cost_activity_class != 0 or cost_action_class != 0 or cost_group != 0 or cost_size != 0, "all costs cant be 0"

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
        out_attw = outputs['attention_logits']  # B, n_max, num_queries

        # target labels and assignments
        tgt_activity_ids, mask_ids = targets[0].decompose()  # B, num_group_max
        tgt_one_hot, mask_one_hot = targets[1].decompose()  # B, n_max, num_groups_max

        valid_areas_ids = crop_to_original(mask_ids)  # batch size, 2 (min, max)
        valid_areas_one_hot = crop_to_original(mask_one_hot)  # batch size, 4 (ymin ymax xmin xmax)

        indices = []
        for b in range(bs):
            tgt_activity_ids_b = tgt_activity_ids[b][valid_areas_ids[b][0]: valid_areas_ids[b][1]]  # [num_group]
            tgt_activity_ids_b_oh = F.one_hot(tgt_activity_ids_b, num_activity_classes)  # num_group, num_activity_cls
            tgt_one_hot_b = tgt_one_hot[b][valid_areas_one_hot[b][0]: valid_areas_one_hot[b][1], valid_areas_one_hot[b][2]: valid_areas_one_hot[b][3]]  # n_persons, n_groups
            n_group = len(tgt_activity_ids_b)
            n_person = tgt_one_hot_b.shape[0]
            out_activity_prob_b = out_activity_prob[b]  # [num_queries, num_classes]
            out_attw_b = out_attw[b][0: n_person, :]  # [num_persons, num_queries]

            tgt_one_hot_b = tgt_one_hot_b.T  # num_groups, n_persons
            out_attw_b = out_attw_b.T  # num_queries, n_persons

            # tgt_size = tgt_one_hot_b.sum(dim=-1)
            # out_size = out_attw_b.sum(dim=-1)

            # P_qn = out_attw_b.clamp_min(1e-6)
            # logP_qn = P_qn.log()
            logP_qn = F.log_softmax(out_attw_b, dim=0)  # num_queries, n_persons
            P_qn = logP_qn.exp()

            P_qn_clamped = P_qn.clamp(min=1e-6, max=1 - 1e-6)
            lognegP_qn = torch.log1p(-P_qn_clamped)  # num_queries, n_persons

            grouping_cost = torch.zeros(num_queries, n_group, device=out_attw.device)
            activity_cost = torch.zeros(num_queries, n_group, device=out_attw.device)
            size_cost = torch.zeros(num_queries, n_group, device=out_attw.device)

            pred_size = P_qn.sum(dim=1)   # num_queries

            for j in range(n_group):
                members = tgt_one_hot_b[j].bool()  # tgt_one_hot_b[j] [n_persons]
                nonmembers = ~tgt_one_hot_b[j].bool()
                m = int(members.sum().item())
                if m == 0:
                    grouping_cost[:, j] = 1e6
                    size_cost[:, j] = 1e6
                else:  # grouping_cost[:, j]: num_queries; logP_qn: [num_queries, n_persons]
                    # grouping_cost[:, j] = -logP_qn[:, members].mean(dim=1)  # cost between each query and a certain group, members are the ground truth of this certain group
                    grouping_cost[:, j] = -(logP_qn[:, members]).mean(dim=1) - (lognegP_qn[:, nonmembers]).mean(dim=1)
                    size_cost[:, j] = (pred_size - float(m)).abs() / max(float(n_person), 1.0)
                activity_cost[:, j] = F.cross_entropy(
                    out_activity_prob_b,  # [num_queries, num_classes]
                    tgt_activity_ids_b[j].expand(num_queries),  # [num_queries]
                    reduction="none"
                )

            # for i, out_attw_b_query in enumerate(out_attw_b):  # n_persons (can be regarded as cls) for certain query: multi cls classification for group
            #     for j, tgt_one_hot_b_group in enumerate(tgt_one_hot_b):  # n_persons (can be regarded as cls) for certain group
            #         grouping_cost[i][j] = torch.norm(out_attw_b_query.float() - tgt_one_hot_b_group.float(), p=2)
            #         # grouping_cost[i][j] = F.binary_cross_entropy_with_logits(out_attw_b_query.float(), tgt_one_hot_b_group.float())  # direction: -> smaller cost  1.  multi cls(persons) classification for groups
            #
            #         # activity_cost[i][j] = F.cross_entropy(out_activity_prob_b[i].float(), tgt_activity_ids_b[j])
            #         # activity_cost[i][j] = -out_activity_prob_b[i].float() * tgt_activity_ids_b[j] - (1 - out_activity_prob_b[i].float()) * (1 - tgt_activity_ids_b[j])  # direction: -> smaller cost  0.
            #         activity_cost[i][j] = F.binary_cross_entropy_with_logits(out_activity_prob_b[i].float(), tgt_activity_ids_b_oh[j].float())

            cost_b = self.cost_group * grouping_cost + self.cost_activity_class * activity_cost + self.cost_size * size_cost
            cost_b = cost_b.cpu().numpy()
            indices_b = linear_sum_assignment(cost_b)
            indices.append(indices_b)  # indices: B, 2 (prediction_id, target_id), num_groups
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  # list with bs length, each element: (src id, tgt id)


def build_matcher(args):
    # TODO: ADD cost_action
    return HungarianMatcher(cost_activity_class=args.set_cost_activity_class, cost_action_class=args.set_cost_action_class, cost_group=args.set_cost_group, cost_size=args.set_cost_size)
