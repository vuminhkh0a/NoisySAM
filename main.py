import time
import numpy as np
import torch
import cv2
import scipy.io
from scipy import ndimage

from data import *
from metrics import *
from noise import *
from model import *


DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

DATASETS = [
    "VOC2012",
    "BSDS500",
    "stanford-background"
]

MODELS = [
    "sam_h",
    "sam2_h",
    "mobile_sam",
]

NOISES = {
    "none": None,
    "flip": flip_transform,
    "rotation": rotation_transform,
    "scale": scale_transform,
    "noise injection": noise_injection_transform,
    "color": color_transform,
    "contrast": contrast_transform,
    "sharpen": sharpen_transform,
    "translation": translation_transform,
}


def get_box_prompts(mask, expand_ratio=0.02):
    
    H, W = mask.shape
    expand_x = int(W * expand_ratio)
    expand_y = int(H * expand_ratio)

    boxes = []

    classes = np.unique(mask)

    for cls in classes:
        if cls:
            binary = (mask == cls).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(binary)

            for i in range(1, num_labels):
                component = (labels == i).astype(np.uint8)

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

def get_point_prompts(mask):
    labeled_mask, num_regions = ndimage.label(mask == 1)
    points = []
    for region_id in range(1, num_regions + 1):
        region = labeled_mask == region_id
        dist = ndimage.distance_transform_edt(region)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        points.append([cx, cy])

    return np.array(points)

def split_masks(y):
    masks = []

    ids = np.unique(y)

    for gid in ids:
        masks.append(y == gid)

    return masks

def sam_predict_mask(predictor, image, prompts, prompt_type):

    if prompt_type == 'point':
        labels = np.ones(len(prompts))
        masks, scores, _ = predictor.predict(
            point_coords=prompts,
            point_labels=labels,
            multimask_output=False
        )
        return masks[0]

    elif prompt_type == 'box':
        final_mask = None

        for box in prompts:
            box = np.array(box)
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=False
            )

            pred = masks[0]

            if final_mask is None:
                final_mask = pred
            else:
                final_mask = np.logical_or(final_mask, pred)

        return final_mask



def evaluate_sam(image, gt_mask, predictor, prompt_type):

    masks = split_masks(gt_mask)

    ious = []
    dices = []
    precisions = []
    recalls = []


    predictor.set_image(image)

    for mask in masks:

        if prompt_type == 'point':
            points = get_point_prompts(mask)
            pred = sam_predict_mask(predictor, image, points, prompt_type)

        elif prompt_type == 'box':
            boxes = get_box_prompts(mask)
            pred = sam_predict_mask(predictor, image, boxes, prompt_type)

        metrics = compute_metrics(pred, mask)

        ious.append(metrics["iou"])
        dices.append(metrics["dice"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])

    mean_metrics = {
        "miou": np.round(np.mean(ious), 2),
        "mdice": np.round(np.mean(dices), 2),
        "mprecision": np.round(np.mean(precisions), 2),
        "mrecall": np.round(np.mean(recalls), 2)
    }

    return mean_metrics


def main():

    start_time = time.perf_counter()

    predictors = get_predictors(MODELS, DEVICE)

    for dataset_name in DATASETS:
        x, y = get_dataset(dataset_name, True)
        for noise_name, transform in NOISES.items():
            

            # for i in range(len(x)):
            for i in range(1):
                image = cv2.imread(x[i])

                if dataset_name == "BSDS500":

                    mask = scipy.io.loadmat(y[i])
                    mask = mask['groundTruth'][0][0]['Segmentation'][0][0]

                else:

                    mask = cv2.imread(y[i], cv2.IMREAD_GRAYSCALE)

                if transform is not None:

                    augmented = transform(image=image, mask=mask)
                    image = augmented["image"]
                    mask = augmented["mask"]
                
                miou_list = []
                mdice_list = []
                mprecision_list = []
                mrecall_list = []

                for model_name in MODELS:

                    predictor = predictors[model_name]

                    metrics = evaluate_sam(image, mask, predictor, 'box')

                    miou_list.append(metrics["miou"])
                    mdice_list.append(metrics["mdice"])
                    mprecision_list.append(metrics["mprecision"])
                    mrecall_list.append(metrics["mrecall"])

            results = {
                "Iou": np.round(np.mean(miou_list), 4).item(),
                "Dice": np.round(np.mean(mdice_list), 4).item(),
                "Precision": np.round(np.mean(mprecision_list), 4).item(),
                "Recall": np.round(np.mean(mrecall_list), 4).item()
            }

            print(
                f"Noise: {noise_name} | Model: {model_name} | Dataset: {dataset_name} | {results}"
            )

            torch.cuda.empty_cache()
        print('\n')

    end_time = time.perf_counter()

    print(f"\nTotal time: {end_time-start_time:.2f} seconds")



if __name__ == "__main__":
    main()
