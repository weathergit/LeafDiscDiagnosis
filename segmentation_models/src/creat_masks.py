import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as seg
from dataloader.DownyData import DownyData
from dataloader.data_aug import get_val_aug, get_processing

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_mask_array(image_dir, masks_dir, models_weights, out_fld):
    CLASSES = ['disc', 'downy']
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    preprocess_fn = seg.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    test_datasets = DownyData(images_dir=image_dir, masks_dir=masks_dir, augmentation=get_val_aug(),
                              preprocessing=get_processing(preprocess_fn), classes=CLASSES)
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=8)
    image_names = test_datasets.ids

    for weight in models_weights:
        model_name = os.path.basename(weight).split('_')[0]
        print(model_name)
        best_model = torch.load(weight, map_location=device)
        for batch, name in zip(test_loader, image_names):
            img, mask = batch
            pr_mask = best_model.predict(img.to(device))
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            outdir = out_fld + model_name + '/'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            np.save(outdir + name[:-4], pr_mask)
        torch.cuda.empty_cache()


def create_mask_array_main():
    # models_weights = sorted(glob.glob('./logs/with_ms_logs/*pth'))
    # image_dir = '../datasets/Seg_dataset/test/images/'
    # masks_dir = '../datasets/Seg_dataset/test/masks/'
    # out_fld = './logs/with_ms_masks/'
    models_weights = sorted(glob.glob('./logs/without_ms_logs/*pth'))
    image_dir = '../datasets/Origin_dataset/test/images/'
    masks_dir = '../datasets/Origin_dataset/test/masks/'
    out_fld = './logs/without_ms_masks/'
    create_mask_array(image_dir, masks_dir, models_weights, out_fld)


def create_painted_mask(image_path, mask_path, alpha=0.5):
    # color: green , red, Imagecolor.getrgb
    colors = [(0, 128, 0), (255, 0, 0)]
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    masks = np.load(mask_path).astype('bool')
    img_to_draw = img.copy()
    for mask, color in zip(masks, colors):
        img_to_draw[mask, :] = color[::-1]
    painted = img * (1 - alpha) + img_to_draw * alpha
    painted = painted.astype('uint8')
    out_path = mask_path.replace('.npy', '.jpg')
    cv2.imwrite(out_path, painted)


def create_ground_truth_mask():
    colors = [(0, 128, 0), (255, 0, 0)]
    alpha = 0.5
    out_dir = './logs/ground_mask/'
    img_folder = "../datasets/Seg_dataset/test/images/"
    mask_folder = "../datasets/Seg_dataset/test/masks/"
    img_paths = sorted(glob.glob(img_folder + '*'))
    mask_paths = sorted(glob.glob(mask_folder + '*'))
    for img_path, msk_path in zip(img_paths, mask_paths):
        # print(img_path, msk_path)
        img = cv2.imread(img_path)
        img_to_draw = img.copy()
        mask = np.load(msk_path)
        mask = [(mask == v) for v in [1, 2]]
        mask = np.stack(mask, axis=0).astype('bool')
        for msk, color in zip(mask, colors):
            img_to_draw[msk, :] = color[::-1]
        painted = img * (1 - alpha) + img_to_draw * alpha
        painted = painted.astype('uint8')
        painted = cv2.resize(painted, (224, 224))
        basename = os.path.basename(img_path)
        out_path = out_dir + basename
        cv2.imwrite(out_path, painted)


def create_painted_mask_main():
    img_folder = "../datasets/Origin_dataset/test/images/"
    mask_folder = './logs/without_ms_masks/'
    img_paths = sorted(glob.glob(img_folder + '*'))
    models = os.listdir(mask_folder)
    for model in models:
        print(model)
        pattern = mask_folder + model + '/*.npy'
        mask_paths = sorted(glob.glob(pattern))
        for i_path, m_path in zip(img_paths, mask_paths):
            create_painted_mask(i_path, m_path, alpha=0.5)


def mv_target_images():
    target = pd.read_csv('./logs/biggest_diff_imgpath.csv')
    # name_list = ["16-1-1-1(20)_17.jpg", "16-1-1-1(40)1_8.jpg",
    #              "IMG_2203_17.jpg","IMG_2217_2.jpg", "IMG_2317_5.jpg",
    #              "IMG_8566_13.jpg", "IMG_8577_16.jpg", "IMG_8656_8.jpg",
    #              "IMG_9002_6.jpg", "IMG_9010_11.jpg"]
    name_list = target['Path'].tolist()
    print(name_list)
    test_folders = ['../datasets/Origin_dataset/test/images/', '../datasets/Seg_dataset/test/images/']
    orig_dst = {}
    for name in name_list:
        orig0 = test_folders[0] + name
        orig1 = test_folders[1] + name
        orig0_dst = orig0.replace('.jpg', '_origin.jpg').replace(test_folders[0], '')
        orig1_dst = orig1.replace('.jpg', '_seg.jpg').replace(test_folders[1], '')
        orig_dst[orig0] = orig0_dst
        orig_dst[orig1] = orig1_dst
    mask_folder = ['./logs/with_ms_masks/', './logs/without_ms_masks/']
    labels = ['with', 'without']
    for label, folder in zip(labels, mask_folder):
        for name in name_list:
            img_list = glob.glob(folder + '*/' + name)
            for fn in img_list:
                dst = fn.replace(folder, '').replace('/', '_' + label + '_')
                orig_dst[fn] = dst
    out_dir = './logs/compare_masks/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for src, dst in orig_dst.items():
        new_dst = out_dir + dst
        print(new_dst)
        shutil.copy(src, new_dst)


if __name__ == '__main__':
    # create_mask_array_main()
    # create_painted_mask_main()
    # mv_target_images()
    create_ground_truth_mask()