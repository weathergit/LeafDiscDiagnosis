# encoding : utf-8
import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np


class DownyData(Dataset):
    CLASSES = ['background', 'disc', 'downy']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = sorted(os.listdir(images_dir))
        self.mds = sorted(os.listdir(masks_dir))
        self.images_fps = [os.path.join(images_dir, images_id) for images_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, masks_id) for masks_id in self.mds]
        self.classes = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.severity_class = [-1, 0.00001, 0.05, 0.25, 0.5, 0.75, 1]
        self.severity_level = [0, 1, 3, 5, 7, 9]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    # @staticmethod
    # def map_to_range(value):
    #     return {
    #         value <= 0.: 0,
    #         0 < value <= 0.05: 1,
    #         0.05 < value <= 0.25: 2,
    #         0.25 < value <= 0.5: 3,
    #         0.5 < value <= 0.75: 4,
    #         0.75 < value <= 1: 5
    #     }[True]

    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.load(self.masks_fps[i])

        masks = [(mask == v) for v in self.classes]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)



