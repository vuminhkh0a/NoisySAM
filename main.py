import time
import numpy as np
import torch
import cv2
import scipy.io

from data import *
from metrics import *
from noise import *
from model import *


DEVICE = "cuda:2"


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


def split_masks(y):
    masks = []

    ids = np.unique(y)

    for gid in ids:
        masks.append(y == gid)

    return masks

def sample_points(mask):
    labeled_mask, num_regions = ndimage.label(mask)
    points = []
    for region_id in range(1, num_regions + 1):
        region = labeled_mask == region_id
        dist = ndimage.distance_transform_edt(region)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        points.append([cx, cy])

    return np.array(points)

def sam_predict_mask(predictor, image, points):

    labels = np.ones(len(points))

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False
    )

    return masks[0]

def evaluate_sam(image, gt_mask, predictor):

    masks = split_masks(gt_mask)

    ious = []
    dices = []
    precisions = []
    recalls = []

    preds = []

    predictor.set_image(image)

    for mask in masks:

        points = sample_points(mask)

        if points is None:
            continue

        pred = sam_predict_mask(predictor, image, points)
        preds.append(pred)

        metrics = compute_metrics(pred, mask)

        ious.append(metrics["iou"])
        dices.append(metrics["dice"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])

    if len(ious) == 0:
        return None

    mean_metrics = {
        "miou": np.mean(ious),
        "mdice": np.mean(dices),
        "mprecision": np.mean(precisions),
        "mrecall": np.mean(recalls)
    }

    return mean_metrics, preds, masks


def main():

    start_time = time.time()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictors = get_predictor(MODELS, DEVICE)

    for noise_name, transform in NOISES.items():

        for model_name in MODELS:

            predictor = predictors[model_name]

            for dataset_name in DATASETS:

                x, y = get_dataset(dataset_name, True)

                miou_list = []
                mdice_list = []
                mprecision_list = []
                mrecall_list = []

                for i in range(len(x)):

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

                    result = evaluate_sam(image, mask, predictor)

                    if result is None:
                        continue

                    metrics, preds, masks = result

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
        print('\n')

    end_time = time.time()

    print(f"\nTotal time: {end_time-start_time:.2f} seconds")



if __name__ == "__main__":
    main()
