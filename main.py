import time
import numpy as np
import torch
import cv2
import sys
import json

from data import *
from metrics import *
from noise import *
from model import *

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODELS = [
    "sam1",
    "sam2",
    "sam3",
    "fastsam",
    "mobilesam",
]

NOISES = {
    "none": None,
    "gaussian_noise": gaussian_noise,
    "motion_blur": motion_blur,
    "snow": snow,
    "brightness": brightness,
    "contrast": contrast,
    "jpeg": jpeg,
}


# ==============================
# Utils
# ==============================
def split_masks(y):
    ids = np.unique(y)
    return [(y == gid) for gid in ids if gid != 0]


def get_box_prompts(mask, expand_ratio=0.02):
    H, W = mask.shape
    expand_x = int(W * expand_ratio)
    expand_y = int(H * expand_ratio)

    boxes = []

    binary = mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)

    for i in range(1, num_labels):
        component = (labels == i)

        ys, xs = np.where(component)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(W - 1, x2 + expand_x)
        y2 = min(H - 1, y2 + expand_y)

        boxes.append([x1, y1, x2, y2])

    return boxes


# ==============================
# Core Evaluation
# ==============================


def evaluate_sam(gt_mask, predictor):

    ious, dices, precisions, recalls, hd95s = [], [], [], [], []

    masks = split_masks(gt_mask)

    for mask in masks:

        boxes = get_box_prompts(mask)

        if len(boxes) == 0:
            pred = np.zeros_like(mask, dtype=bool)

        else:
            merged_pred = None

            for b in boxes:
                p, _, _ = predictor.predict(
                    box=np.array(b),
                    multimask_output=False
                )

                if p is None or len(p) == 0:
                    continue

                p = p[0]  # (H, W)

                if merged_pred is None:
                    merged_pred = p
                else:
                    merged_pred = np.logical_or(merged_pred, p)

            if merged_pred is None:
                pred = np.zeros_like(mask, dtype=bool)
            else:
                pred = merged_pred

        metrics = compute_metrics(pred, mask)

        ious.append(metrics["iou"])
        dices.append(metrics["dice"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        hd95s.append(metrics["hd95"])

    return {
        "miou": float(np.mean(ious)) if len(ious) > 0 else 0.0,
        "mdice": float(np.mean(dices)) if len(dices) > 0 else 0.0,
        "mprecision": float(np.mean(precisions)) if len(precisions) > 0 else 0.0,
        "mrecall": float(np.mean(recalls)) if len(recalls) > 0 else 0.0,
        "mhd95": float(np.mean(hd95s)) if len(hd95s) > 0 else 0.0,
    }

# ==============================
# Main
# ==============================
def main():
    
    output_results = []
    # Load models
    predictors = get_predictors(MODELS, DEVICE)
    print("Models:", predictors.keys())
    print("Device:", DEVICE)
    sys.stdout.flush()

    # Load dataset
    x_path, y_path = get_VOC2012()
    N = len(x_path)

    print("Dataset size:", N)

    for noise_name, transform in NOISES.items():

        for severity in range(1, 6):

            if transform is None and severity > 1:
                continue

            for model_name in MODELS:
                start_time = time.perf_counter()

                predictor = predictors[model_name]

                miou_list = []
                mdice_list = []
                mprecision_list = []
                mrecall_list = []
                mhd95_list = []

                for i, (xp, yp) in enumerate(zip(x_path, y_path)):
                    
                    # ===== Load image =====
                    image = cv2.imread(xp)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.ascontiguousarray(image)

                    mask = cv2.imread(yp, cv2.IMREAD_GRAYSCALE)
                    mask = np.where(mask != 220, mask, 0)

                    # ===== Apply noise =====
                    if transform is not None:
                        image = transform(image, severity)

                    predictor.set_image(image)

                    # ===== Evaluate =====
                    metrics = evaluate_sam(mask, predictor)

                    miou_list.append(metrics["miou"])
                    mdice_list.append(metrics["mdice"])
                    mprecision_list.append(metrics["mprecision"])
                    mrecall_list.append(metrics["mrecall"])
                    mhd95_list.append(metrics["mhd95"])

                results = {
                    "Iou": float(np.round(np.mean(miou_list), 2)),
                    "Dice": float(np.round(np.mean(mdice_list), 2)),
                    "Precision": float(np.round(np.mean(mprecision_list), 2)),
                    "Recall": float(np.round(np.mean(mrecall_list), 2)),
                    "HD95": float(np.round(np.mean(mhd95_list), 2)),
                }

                print(f"Noise: {noise_name} | Severity: {severity} | Model: {model_name} | Results: {results}")
                end_time = time.perf_counter()
                print(f"Time: {end_time - start_time:.2f} seconds")
                print("----------------------------------------")

                output_results.append({
                    "noise": noise_name,
                    "severity": severity,
                    "model": model_name,
                    "metrics": results,
                })

                sys.stdout.flush()
    
    with open("results.json", "w") as f:
        json.dump(output_results, f, indent=4)

    print("Saved results to results.json")

if __name__ == "__main__":
    main()