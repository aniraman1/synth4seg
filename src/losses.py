"""Permutation-invariant losses for segmenting a variable number of structures.

The network predicts K channels (K >= the largest number of structures in any example) and
a softmax over them assigns each voxel to one "slot". Which slot a structure lands in is
arbitrary, so the loss must not depend on channel order. The co-assignment loss compares
*whether two voxels share a slot* with *whether they belong to the same structure*:

    pred   coassign(i, j) = P_i . P_j        (P_i = softmax probabilities at voxel i)
    target coassign(i, j) = [y_i == y_j]

Instead of sampling voxel pairs, the loss is computed exactly over all pairs by grouping
them into blocks by ground-truth structure. For structures a and b the mean predicted
co-assignment over all voxel pairs (i in a, j in b) is

    S_ab = mean_{i in a, j in b} P_i . P_j = m_a . m_b,     m_a = mean_{i in a} P_i

so only the (C, K) matrix of per-structure mean probabilities is needed: O(N K) time and
memory, no N x N matrix. S_aa should be 1 (structure a uses one slot everywhere) and S_ab
(a != b) should be 0 (different structures never share a slot). Every structure counts
equally, so a thin tree weighs as much as the background.

    from src.dataset import SynthDataset
    from src.losses import coassignment_loss, matched_dice

    ds = SynthDataset(batch_size=2, target="labels", n_blobs=(2, 12), n_trees=(0, 6))
    model = UNet(out_channels=ds.n_label_classes + 4)       # a few spare slots
    for image, labels in ds:
        logits = model(image)                               # (B, K, S, S, S)
        loss = coassignment_loss(logits, labels)
        ...
        pred = logits.argmax(1)                             # instance map, arbitrary ids
        dice = matched_dice(pred, labels)                   # (B,) after optimal matching
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def structure_means(probs, labels, ignore_index=None):
    """Per-structure mean probability vectors.

    Args:
        probs: (B, K, *spatial) probabilities (softmax over K).
        labels: (B, *spatial) integer structure ids, any numbering (ids need not be
            contiguous; absent ids are flagged in `present`).
        ignore_index: label value whose voxels are left out.

    Returns:
        means: (B, C, K) float32, row a = mean of probs over the voxels of structure a.
        present: (B, C) bool, which ids occur in each example.
    """
    B, K = probs.shape[:2]
    p = probs.reshape(B, K, -1).float()
    y = labels.reshape(B, -1).long()
    if ignore_index is not None:
        keep = y != ignore_index
        y = torch.where(keep, y, torch.zeros_like(y))
        p = p * keep[:, None]
    else:
        keep = None
    C = int(y.max()) + 1                                # one host sync per call
    if keep is not None:
        y = torch.where(keep, y, torch.full_like(y, C))  # ignored voxels -> dropped column
    Cx = C + (keep is not None)
    sums = p.new_zeros(B, K, Cx).scatter_add_(2, y[:, None].expand(-1, K, -1), p)
    counts = p.new_zeros(B, Cx).scatter_add_(1, y, torch.ones_like(y, dtype=p.dtype))
    sums, counts = sums[..., :C], counts[:, :C]
    present = counts > 0
    means = sums.transpose(1, 2) / counts.clamp_min(1)[..., None]
    return means, present


def coassignment_loss(logits, labels, neg_weight=1.0, ignore_index=None, eps=1e-6,
                      return_parts=False):
    """Exact all-pairs co-assignment loss (permutation invariant over channels).

    BCE between the mean predicted co-assignment S_ab of every pair of structures (a, b)
    and its target (1 if a == b else 0):

        loss = mean_a [ -log S_aa  +  neg_weight * sum_{b != a} -log(1 - S_ab) ]

    averaged over the batch. The negative term is summed over the other structures, not
    averaged over all pairs: averaging dilutes the penalty for two structures sharing a
    slot as the number of structures grows, and optimisation then gets stuck with merged
    structures. At the minimum each structure is assigned to one channel with
    probability 1 everywhere and no two structures share a channel; channels left over
    stay empty.

    Args:
        logits: (B, K, *spatial) raw network output. Softmax is taken over K in float32.
            K must be >= the number of structures in every example, or the loss cannot
            reach zero.
        labels: (B, *spatial) integer structure ids (e.g. SynthDataset target="labels"
            or the raw SynthGenerator labels; the numbering does not matter).
        neg_weight: weight of the "different structures must not share a slot" term.
            Lowering it much below 1 makes merged structures more likely.
        ignore_index: label value to exclude from the loss.
        eps: clamp for the logs.
        return_parts: also return a dict with the positive and negative terms (detached).
    """
    probs = logits.float().softmax(1)
    means, present = structure_means(probs, labels, ignore_index)
    S = means @ means.transpose(1, 2)                   # (B, C, C)
    S = S.clamp(eps, 1 - eps)
    C = S.shape[1]
    eye = torch.eye(C, dtype=torch.bool, device=S.device)
    pair = present[:, :, None] & present[:, None, :]
    pos_mask = pair & eye
    neg_mask = pair & ~eye
    n_struct = pos_mask.sum((1, 2)).clamp_min(1)
    pos = (-torch.log(S) * pos_mask).sum((1, 2)) / n_struct
    neg = (-torch.log1p(-S) * neg_mask).sum((1, 2)) / n_struct
    loss = (pos + neg_weight * neg).mean()
    if return_parts:
        return loss, {"pos": pos.mean().detach(), "neg": neg.mean().detach()}
    return loss


class CoassignmentLoss(nn.Module):
    """nn.Module wrapper around `coassignment_loss`."""

    def __init__(self, neg_weight=1.0, ignore_index=None, eps=1e-6):
        super().__init__()
        self.neg_weight, self.ignore_index, self.eps = neg_weight, ignore_index, eps

    def forward(self, logits, labels):
        return coassignment_loss(logits, labels, self.neg_weight, self.ignore_index,
                                 self.eps)


@torch.no_grad()
def match_channels(pred, labels):
    """Optimal one-to-one matching of predicted ids to ground-truth ids by Dice.

    Args:
        pred: (B, *spatial) predicted ids, e.g. logits.argmax(1).
        labels: (B, *spatial) ground-truth ids.

    Returns:
        A list with one dict per example: {"gt": array of ground-truth ids present,
        "pred": matched predicted id for each (-1 if unmatched), "dice": Dice of each
        ground-truth structure with its match (0 if unmatched), "n_pred": number of
        non-empty predicted ids}.
    """
    B = pred.shape[0]
    p = pred.reshape(B, -1).long()
    y = labels.reshape(B, -1).long()
    K = int(p.max()) + 1
    C = int(y.max()) + 1
    out = []
    for b in range(B):
        conf = torch.bincount(y[b] * K + p[b], minlength=C * K).view(C, K).double()
        n_gt, n_pr = conf.sum(1), conf.sum(0)
        dice = (2 * conf / (n_gt[:, None] + n_pr[None, :]).clamp_min(1)).cpu().numpy()
        gt_ids = np.flatnonzero(n_gt.cpu().numpy() > 0)
        pr_ids = np.flatnonzero(n_pr.cpu().numpy() > 0)
        d = dice[np.ix_(gt_ids, pr_ids)]
        rows, cols = linear_sum_assignment(-d)
        match = np.full(len(gt_ids), -1)
        score = np.zeros(len(gt_ids))
        match[rows] = pr_ids[cols]
        score[rows] = d[rows, cols]
        out.append({"gt": gt_ids, "pred": match, "dice": score, "n_pred": len(pr_ids)})
    return out


def matched_dice(pred, labels):
    """(B,) mean Dice over ground-truth structures after optimal matching."""
    return torch.tensor([m["dice"].mean() for m in match_channels(pred, labels)])
