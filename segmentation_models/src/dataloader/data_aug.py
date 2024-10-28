# encoding : utf-8

import albumentations as albu


def get_train_aug():
    train_transform = albu.Compose([albu.Resize(height=224, width=224, always_apply=True),
                                    albu.HorizontalFlip(p=0.5),
                                    albu.HueSaturationValue(p=1)])
    return train_transform


def get_val_aug():
    return albu.Compose([albu.Resize(height=224, width=224, always_apply=True)])


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_processing(preprocessing_fn):
    _process=[albu.Lambda(image=preprocessing_fn),
              albu.Lambda(image=to_tensor, mask=to_tensor)]
    return albu.Compose(_process)

