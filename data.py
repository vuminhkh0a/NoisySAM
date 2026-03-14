from noise import *
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import os

IMAGE_SIZE = 256
VOC_PATH = '/mnt/nvme0/home/utbt/KhoaVM/NoisySAM/data/VOC2012'
BSDS500_PATH = '/mnt/nvme0/home/utbt/KhoaVM/NoisySAM/data/BSDS500'
SB_PATH = '/mnt/nvme0/home/utbt/KhoaVM/NoisySAM/data/stanford-bg'


class Custom_Dataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_path = self.images[i]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        mask_path = self.masks[i]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        return np.astype(image, np.float32), np.astype(mask, np.float32)
    


    
def get_VOC2012():
    image_paths = os.path.join(VOC_PATH, 'JPEGImages')
    mask_paths = os.path.join(VOC_PATH, 'SegmentationObject')

    img_path_lst = []
    mask_path_lst = []

    with open('/mnt/nvme0/home/utbt/KhoaVM/NoisySAM/data/VOC2012/ImageSets/Segmentation/train.txt', 'r') as f:
        for line in f:
            img_path_lst.append(line.strip() + '.jpg')
            mask_path_lst.append(line.strip() + '.png')

    with open('/mnt/nvme0/home/utbt/KhoaVM/NoisySAM/data/VOC2012/ImageSets/Segmentation/trainval.txt', 'r') as f:
        for line in f:
            img_path_lst.append(line.strip() + '.jpg')
            mask_path_lst.append(line.strip() + '.png')

    with open('/mnt/nvme0/home/utbt/KhoaVM/NoisySAM/data/VOC2012/ImageSets/Segmentation/val.txt', 'r') as f:
        for line in f:
            img_path_lst.append(line.strip() + '.jpg')
            mask_path_lst.append(line.strip() + '.png')

    x = sorted([os.path.join(image_paths, image_path_i) for image_path_i in img_path_lst])
    y = sorted([os.path.join(mask_paths, mask_path_i) for mask_path_i in mask_path_lst])
    return x, y

def get_BSDS500():
    image_paths = os.path.join(BSDS500_PATH, 'images')
    mask_paths = os.path.join(BSDS500_PATH, 'ground_truth')

    img_path_lst = []
    mask_path_lst = []
    for ty in ['test', 'train', 'val']:
        for img_i, mask_i in zip(sorted(os.listdir(os.path.join(image_paths, ty))), sorted(os.listdir(os.path.join(mask_paths, ty)))):
            img_path_lst.append(os.path.join(image_paths, ty, img_i))
            mask_path_lst.append(os.path.join(mask_paths, ty, mask_i))

    x, y = img_path_lst, mask_path_lst

    return x, y


def get_stanford_background():
    img_path_lst = []
    mask_path_lst = []

    for img_i, mask_i in zip(sorted(os.listdir(os.path.join(SB_PATH, 'images'))), sorted(os.listdir(os.path.join(SB_PATH, 'labels_colored')))):
        img_path_lst.append(os.path.join(SB_PATH, 'images', img_i))
        mask_path_lst.append(os.path.join(SB_PATH, 'labels_colored', mask_i))

    x = img_path_lst
    y = mask_path_lst

    return x, y

def get_dataset(dataset_name, path_only):
    get_path = {'VOC2012':get_VOC2012, 'BSDS500': get_BSDS500, 'stanford-background': get_stanford_background}
    x, y = get_path[dataset_name]()
    if path_only:
        return x, y
    return Custom_Dataset(x, y)


def get_loader(dataset_name, batch_size, num_workers, pin_memory):
    dataset = get_dataset(dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return loader

