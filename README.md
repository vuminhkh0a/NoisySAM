# NoisySAM - Evaluate foundation model robustness under perturbations for natural image segmentation [Ongoing]

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
```

## Future Extensions

### Additional Models

The following segmentation foundation models are planned to be integrated:

- SAM3
- MobileSAMv2
- FastSAM

These models will be evaluated under the same noise and perturbation settings to provide a consistent robustness comparison.

### Additional Datasets

Future experiments will also include more complex and large-scale datasets:

- COCO
- Cityscapes

These datasets introduce more diverse scenes, object categories, and challenging segmentation scenarios, enabling a more comprehensive robustness benchmark.

### Additional perturbations
- **MixUp**
- **CutMix**
- **CutOut**

### Other Planned Improvements

- Testing on medical benchmark dataset using medical foundation models such as MedSAM
- Using other prompt strateges such box or text
- Visualization tools for prediction comparison across models
