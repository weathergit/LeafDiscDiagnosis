import  os
import shutil
import glob
import numpy as np


def yolo_split():
    images = sorted(glob.glob('../origin_data/images/*'))
    labels = sorted(glob.glob('../origin_data/labels_txt/*'))

    assert len(images) == len(labels)
    data_number = len(images)
    k1 = int(data_number*0.6)
    k2 = int(data_number*0.2)
    # 20%
    k3 = data_number - k1 - k2

    k1_2 = k1 + k2

    np.random.seed(42)
    np.random.shuffle(images)
    np.random.seed(42)
    np.random.shuffle(labels)

    for i, (ipth, mpth) in enumerate(zip(images, labels)):
        basename_i = os.path.basename(ipth)
        basename_m = os.path.basename(mpth)
        if i <= k1:    
            img_dst = '../datasets/images/train/'
            lbl_dst = '../datasets/labels/train/'
            shutil.copy(ipth, dst=img_dst)
            shutil.copy(mpth, dst=lbl_dst)
        elif k1 < i <= k1_2:
            img_dst = '../datasets/images/val/'
            lbl_dst = '../datasets/labels/val/'
            shutil.copy(ipth, dst=img_dst)
            shutil.copy(mpth, dst=lbl_dst)
        elif i > k1_2:
            img_dst = '../datasets/images/test/'
            lbl_dst = '../datasets/labels/test/'
            shutil.copy(ipth, dst=img_dst)
            shutil.copy(mpth, dst=lbl_dst)


def seg_split():
    images = sorted(glob.glob('../crop_datasets/crop_images/*'))
    masks = sorted(glob.glob('../crop_datasets/crop_masks/*npy'))
    assert len(images) == len(masks)
    data_number = len(images)
    k1 = int(data_number * 0.6)
    k2 = int(data_number * 0.2)
    # 20%
    k3 = data_number - k1 - k2

    k1_2 = k1 + k2

    np.random.seed(42)
    np.random.shuffle(images)
    np.random.seed(42)
    np.random.shuffle(masks)

    for i, (ipth, mpth) in enumerate(zip(images, masks)):
        basename_i = os.path.basename(ipth)
        basename_m = os.path.basename(mpth)
        if i <= k1:
            img_dst = '../crop_datasets/datasets/train/images/'
            lbl_dst = '../crop_datasets/datasets/train/masks/'
            shutil.copy(ipth, dst=img_dst)
            shutil.copy(mpth, dst=lbl_dst)
        elif k1 < i <= k1_2:
            img_dst = '../crop_datasets/datasets/val/images/'
            lbl_dst = '../crop_datasets/datasets/val/masks/'
            shutil.copy(ipth, dst=img_dst)
            shutil.copy(mpth, dst=lbl_dst)
        elif i > k1_2:
            img_dst = '../crop_datasets/datasets/test/images/'
            lbl_dst = '../crop_datasets/datasets/test/masks/'
            shutil.copy(ipth, dst=img_dst)
            shutil.copy(mpth, dst=lbl_dst)


if __name__ == '__main__':
    # yolo_split()
    seg_split()


