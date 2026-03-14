import albumentations as A
import torch
import numpy as np
import cv2

no_transform = A.Compose([])

flip_transform = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
])

rotation_transform = A.Compose([
    A.Rotate(limit=45, p=1.0),
    A.RandomRotate90(p=1.0),
])

scale_transform = A.Compose([
    A.RandomScale(scale_limit=0.3, p=1.0),
])

noise_injection_transform = A.Compose([
    A.GaussNoise(p=1.0),
    A.ISONoise(p=1.0),
])

color_transform = A.Compose([
    A.HueSaturationValue(p=1.0),
    A.RGBShift(p=1.0),
])

contrast_transform = A.Compose([
    A.RandomBrightnessContrast(p=1.0),
    A.RandomGamma(p=1.0),
])

sharpen_transform = A.Compose([
    A.Sharpen(p=1.0),
])

translation_transform = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.0,
        rotate_limit=0,
        p=1.0
    ),
])



def mixup(x1, y1, x2, y2, alpha=0.4):
    if(x1.shape != x2.shape):
        x2 = cv2.resize(x2, (x1.shape[0], x1.shape[1]))
        y2 = cv2.resize(y2, (y1.shape[0], y1.shape[1]))
        
    lam = np.random.beta(alpha, alpha)

    x = lam * x1 + (1 - lam) * x2

    return x.astype(x1.dtype), y1, y2, lam


def cutmix(x1, y1, x2, y2, alpha=1.0):
    if(x1.shape != x2.shape):
        x2 = cv2.resize(x2, (x1.shape[0], x1.shape[1]))
        y2 = cv2.resize(y2, (y1.shape[0], y1.shape[1]))

    H, W, C = x1.shape

    lam = np.random.beta(alpha, alpha)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bw = int(W * np.sqrt(1 - lam))
    bh = int(H * np.sqrt(1 - lam))

    x1_ = max(cx - bw // 2, 0)
    x2_ = min(cx + bw // 2, W)
    y1_ = max(cy - bh // 2, 0)
    y2_ = min(cy + bh // 2, H)

    x = x1.copy()
    y = y1.copy()

    x[y1_:y2_, x1_:x2_] = x2[y1_:y2_, x1_:x2_]
    y[y1_:y2_, x1_:x2_] = y2[y1_:y2_, x1_:x2_]

    lam = 1 - ((x2_ - x1_) * (y2_ - y1_) / (H * W))

    return x, y, lam
    
def cutout(x, y, ratio=0.4, fill_value=0, ignore_label=255):
    H, W, C = x.shape

    cut_w = int(W * ratio)
    cut_h = int(H * ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    x_out = x.copy()
    y_out = y.copy()

    x_out[y1:y2, x1:x2] = fill_value
    y_out[y1:y2, x1:x2] = ignore_label

    return x_out, y_out
