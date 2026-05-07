import cv2
import numpy as np 
import albumentations as A 
import skimage as sk 
from PIL import Image
from scipy.ndimage import zoom as scizoom 
from io import BytesIO


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[:2]

    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))

    top = (h - ch) // 2
    left = (w - cw) // 2

    img = scizoom(
        img[top:top + ch, left:left + cw],
        (zoom_factor, zoom_factor, 1),
        order=1
    )

    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[
        trim_top:trim_top + h,
        trim_left:trim_left + w
    ]

def motion_blur_kernel(size, angle):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1
    kernel /= size

    rot_mat = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, rot_mat, (size, size))

    return kernel
    

# =======================================================================================================================


def gaussian_noise(image, severity:int=1):
    
    temp_image = image

    if severity == 1:
        std_range = (0.00, 0.02)
    elif severity == 2:
        std_range = (0.02, 0.05)
    elif severity == 3:
        std_range = (0.05, 0.10)
    elif severity == 4:
        std_range = (0.10, 0.18)
    elif severity == 5:
        std_range = (0.18, 0.30)
    else:
        raise ValueError("severity must be in [1, 5]")

    transform = A.GaussNoise(std_range=std_range, p=1)
    transformed_image = transform(image=temp_image)['image']
    return transformed_image
    

def motion_blur(image, severity: int = 1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    radius, sigma = c
    angle = np.random.uniform(-45, 45)

    if not isinstance(image, np.ndarray):
        transformed_image = np.array(image)
    else:
        transformed_image = image.copy()

    k = radius
    center = k // 2

    kernel = np.zeros((k, k), dtype=np.float32)

    xs = np.arange(k) - center
    gauss = np.exp(-(xs**2) / (2 * sigma**2))
    gauss /= gauss.sum()

    kernel[center, :] = gauss

    rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (k, k))

    transformed_image = cv2.filter2D(transformed_image, -1, kernel)

    if transformed_image.ndim == 2:  # grayscale
        transformed_image = np.stack([transformed_image]*3, axis=-1)

    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)

    return transformed_image


def snow(image, severity:int=1): 
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
    
    transformed_image = np.array(image, dtype=np.float32) / 255 
    h, w = transformed_image.shape[:2]

    snow_layer = np.random.normal(loc=c[0], scale=c[1], size=(h, w))
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2]).squeeze()
    snow_layer[snow_layer < c[3]] = 0
    kernel = motion_blur_kernel(c[4], angle=np.random.uniform(-135, -45))
    snow_layer = cv2.filter2D(snow_layer, -1, kernel)
    snow_layer = np.clip(snow_layer, 0, 1)
    snow_layer = snow_layer[..., np.newaxis]
    gray = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

    transformed_image = c[6] * transformed_image + (1 - c[6]) * np.maximum(transformed_image, gray * 1.5 + 0.5) 
    transformed_image = transformed_image + snow_layer + np.rot90(snow_layer, k=2) 
    transformed_image = np.clip(transformed_image, 0, 1) * 255 
    return transformed_image.astype(np.uint8)

def brightness(image, severity:int=1): 
    c = [.1, .2, .3, .4, .5][severity - 1]
    transformed_image = image 
    transformed_image = np.array(transformed_image) / 255 
    transformed_image = sk.color.rgb2hsv(transformed_image) 
    transformed_image[:, :, 2] = np.clip(transformed_image[:, :, 2] + c, 0, 1)
    transformed_image = sk.color.hsv2rgb(transformed_image)
    transformed_image = np.clip(transformed_image, 0, 1) * 255
    return transformed_image.astype(np.uint8)

def contrast(image, severity: int=1):
    c = [0.75, 0.6, 0.5, 0.4, 0.3][severity - 1]  

    transformed_image = np.array(image).astype(np.float32) / 255.
    means = np.mean(transformed_image, axis=(0, 1), keepdims=True)

    transformed_image = (transformed_image - means) * c + means
    transformed_image = np.clip(transformed_image, 0, 1) * 255

    return transformed_image.astype(np.uint8)

def jpeg(image, severity:int=1): 
    c = [40, 30, 25, 18, 12][severity - 1]

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = image.astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format='JPEG', quality=c)
    transformed_image = np.array(Image.open(buffer))

    buffer.close()
    return transformed_image  

 


