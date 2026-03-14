import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_metrics(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall
    }

def metrics_dice(target, pred, smooth=1e-15):
    target_f = torch.flatten(target)
    pred_f = torch.flatten(pred)
    intersection = torch.sum(target_f * pred_f)
    return (2. * intersection + smooth) / (torch.sum(target_f) + torch.sum(pred_f) + smooth)

def metrics_iou(target, pred, smooth=1e-15):
    target_f = torch.flatten(target)
    pred_f = torch.flatten(pred)
    intersection = torch.sum(target_f * pred_f)
    union = torch.sum(target_f) + torch.sum(pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def metrics_recall_precision(target, pred):
    tp = torch.sum(target * pred)
    fp = torch.sum(pred) - tp
    fn = torch.sum(target) - tp
    recall = ((tp + 1e-6) / (tp + fn + 1e-6))
    precision = ((tp + 1e-6) / (tp + fp + 1e-6))
    return recall, precision

def metrics_hd95(target, pred, spacing=None):
    pred = pred.bool()
    target = target.bool()

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')
    pred_points = torch.nonzero(pred).float()
    target_points = torch.nonzero(target).float()

    if spacing is not None:
        spacing_tensor = torch.tensor(spacing, device=pred.device).float()
        pred_points = pred_points * spacing_tensor
        target_points = target_points * spacing_tensor

    d_pred_to_target = []
    for batch in pred_points.split(4096):
        d = torch.cdist(batch, target_points)
        min_d, _ = torch.min(d, dim=1)
        d_pred_to_target.append(min_d)
    d_pred_to_target = torch.cat(d_pred_to_target)
    d_target_to_pred = []
    for batch in target_points.split(4096):
        d = torch.cdist(batch, pred_points)
        min_d, _ = torch.min(d, dim=1)
        d_target_to_pred.append(min_d)
    d_target_to_pred = torch.cat(d_target_to_pred)

    hd95_val = max(
        torch.quantile(d_pred_to_target, 0.95).item(),
        torch.quantile(d_target_to_pred, 0.95).item()
    )
    return hd95_val



