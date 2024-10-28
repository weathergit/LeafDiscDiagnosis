import os
from torch.utils.data import DataLoader
import segmentation_models_pytorch as seg
from .DownyData import DownyData
from .data_aug import get_train_aug, get_val_aug, get_processing


ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['disc', 'downy']


def load_data(data_dir, encoder=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=CLASSES, batch_size=64):
    img_train_dir = os.path.join(data_dir, 'train/images/')
    mask_train_dir = os.path.join(data_dir, 'train/masks/')

    img_val_dir = os.path.join(data_dir, 'val/images/')
    mask_val_dir = os.path.join(data_dir, 'val/masks/')

    preprocess_fn = seg.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = DownyData(images_dir=img_train_dir, masks_dir=mask_train_dir,
                              augmentation=get_train_aug(),
                              preprocessing=get_processing(preprocess_fn), classes=classes)

    valid_dataset = DownyData(images_dir=img_val_dir, masks_dir=mask_val_dir,
                              augmentation=get_val_aug(),
                              preprocessing=get_processing(preprocess_fn), classes=classes)
    print('training size:', len(train_dataset))
    print('valid size:', len(valid_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, valid_loader
