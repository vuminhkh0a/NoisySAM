import torch
import numpy as np
import sys
from scipy.spatial import cKDTree
import math

def compute_metrics(pred, gt, device="cuda"):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, device=device)
    else:
        pred = pred.to(device)

    if not torch.is_tensor(gt):
        gt = torch.tensor(gt, device=device)
    else:
        gt = gt.to(device)

    pred = pred.bool()
    gt = gt.bool()

    pred_f = pred.view(-1)
    gt_f = gt.view(-1)

    tp = torch.logical_and(pred_f, gt_f).sum().float()
    fp = torch.logical_and(pred_f, ~gt_f).sum().float()
    fn = torch.logical_and(~pred_f, gt_f).sum().float()

    eps = 1e-6

    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    hd95 = metrics_hd95(gt, pred)

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "hd95": hd95
    }

def _boundary_pts(mask: np.ndarray):
    """
    Lấy boundary pixels của mask
    """
    from scipy import ndimage

    eroded = ndimage.binary_erosion(mask)
    boundary = mask ^ eroded
    pts = np.argwhere(boundary)

    return pts


def metrics_hd95(target, pred, reduction="mean"):
    """
    HD95 theo kiểu boundary + KDTree + percentile(95)

    target, pred:
        (H, W) hoặc (1, H, W) hoặc (B, 1, H, W)
    """

    # --- to numpy ---
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    # --- normalize shape ---
    if pred.ndim == 2:
        pred = pred[None, None, ...]
        target = target[None, None, ...]

    elif pred.ndim == 3:
        pred = pred[None, ...]
        target = target[None, ...]

    B = pred.shape[0]

    results = []

    for b in range(B):
        p = pred[b, 0].astype(bool)
        t = target[b, 0].astype(bool)

        H, W = p.shape
        diag = math.sqrt(H**2 + W**2)

        # --- edge cases ---
        if not p.any() and not t.any():
            results.append(0.0)
            continue

        if not p.any() or not t.any():
            results.append(diag)
            continue

        # --- boundary points ---
        p_pts = _boundary_pts(p)
        t_pts = _boundary_pts(t)

        if len(p_pts) == 0 or len(t_pts) == 0:
            results.append(diag)
            continue

        # --- KDTree ---
        tree_t = cKDTree(t_pts)
        tree_p = cKDTree(p_pts)

        d1, _ = tree_t.query(p_pts)
        d2, _ = tree_p.query(t_pts)

        hd95 = np.percentile(np.concatenate([d1, d2]), 95)

        results.append(float(hd95))

    results = np.array(results)

    if reduction == "mean":
        return float(results.mean())
    elif reduction == "none":
        return results
    else:
        raise ValueError("reduction must be 'mean' or 'none'")
    
# def metrics_hd95(target, pred, spacing=None, reduction="mean"):
#     """
#     target, pred:
#         (H, W) hoặc (1, H, W) hoặc (B, 1, H, W)
#     """

#     # --- ensure tensor ---
#     if not torch.is_tensor(pred):
#         pred = torch.tensor(pred)
#     if not torch.is_tensor(target):
#         target = torch.tensor(target)

#     device = pred.device

#     # --- normalize shape ---
#     if pred.ndim == 2:
#         pred = pred.unsqueeze(0).unsqueeze(0)
#         target = target.unsqueeze(0).unsqueeze(0)

#     elif pred.ndim == 3:
#         pred = pred.unsqueeze(0)
#         target = target.unsqueeze(0)

#     # (B, 1, H, W)
#     B = pred.shape[0]

#     pred = pred.bool()
#     target = target.bool()

#     results = []

#     for b in range(B):
#         p = pred[b, 0]
#         t = target[b, 0]

#         # --- edge cases ---
#         if p.sum() == 0 and t.sum() == 0:
#             results.append(torch.tensor(0.0, device=device))
#             continue
#         if p.sum() == 0 or t.sum() == 0:
#             results.append(torch.tensor(float("inf"), device=device))
#             continue

#         # --- lấy tọa độ ---
#         p_pts = torch.nonzero(p, as_tuple=False).float()
#         t_pts = torch.nonzero(t, as_tuple=False).float()

#         if spacing is not None:
#             spacing_tensor = torch.tensor(spacing, device=device).float()
#             p_pts = p_pts * spacing_tensor
#             t_pts = t_pts * spacing_tensor

#         # --- tính khoảng cách ---
#         def min_dist(A, B):
#             dists = []
#             for batch in A.split(4096):
#                 d = torch.cdist(batch, B)
#                 dists.append(d.min(dim=1)[0])
#             return torch.cat(dists)

#         d_p_t = min_dist(p_pts, t_pts)
#         d_t_p = min_dist(t_pts, p_pts)

#         hd95 = torch.max(
#             torch.quantile(d_p_t, 0.95),
#             torch.quantile(d_t_p, 0.95)
#         )

#         results.append(hd95)

#     results = torch.stack(results)

#     if reduction == "mean":
#         return results.mean().item()
#     elif reduction == "none":
#         return results
#     else:
#         raise ValueError("reduction must be 'mean' or 'none'")


