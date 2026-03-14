# NoisySAM - Foundation Model Robustness for Image Segmentation [Ongoing]

This project evaluates the robustness of foundation segmentation models under different types of noise and image perturbations.

The goal is to systematically test how segmentation performance changes when images are affected by transformations such as geometric distortions, noise injection, color shifts, and mixed-image augmentations.

## Current Models

The following foundation models are currently evaluated:

- SAM
- SAM2
- MobileSAM

## Current Datasets

- BSDS500
- VOC2012
- Stanford Background Dataset

## Noise and Transformations

Various perturbations are applied to the input images using Albumentations:

### Geometric Transformations
- Horizontal + Vertical Flip
- Rotation
- RandomRotate90
- Random Scale
- Translation (Shift)

### Noise Injection
- Gaussian Noise
- ISO Noise

### Color Transformations
- Hue / Saturation Shift
- RGB Shift

### Contrast and Illumination
- Random Brightness / Contrast
- Gamma Adjustment

### Image Enhancement
- Sharpen

### Mixed-image Augmentations
Additional perturbations include:

- **MixUp**
- **CutMix**
- **CutOut**

These operations simulate partial occlusion, mixed visual content, and information loss.

## Inference Strategy

The segmentation models are prompted using sampled points extracted from the ground-truth mask.

Point selection strategy:

1. Compute the centroid of each connected region.
2. Select the centroid as the primary prompt point.
3. Randomly sample additional points from the region.
4. If multiple disconnected regions exist, points are sampled from each region.

Example implementation:

```python
def sample_points(mask, n_points=5):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    points = []

    cx = int(xs.mean())
    cy = int(ys.mean())
    points.append([cx, cy])

    if len(xs) > n_points:
        idx = np.random.choice(len(xs), n_points-1, replace=False)
        for i in idx:
            points.append([xs[i], ys[i]])

    return np.array(points)
