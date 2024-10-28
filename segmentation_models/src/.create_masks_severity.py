import os
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    test_dadasets = DownyData(images_dir=image_dir, masks_dir=masks_dir, augmentation=get_val_aug(),
                              preprocessing=get_processing(preprocess_fn), classes=CLASSES)
    test_loader = DataLoader(test_dadasets, batch_size=1, shuffle=False, num_workers=8)
    image_names = test_dadasets.ids

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


def create_masks_with_mobilesam_or_without():
    # create masks for test set by using models, with Mobile SAM as agent.
    models_weights = sorted(glob.glob('./logs/weights3/*pth'))
    image_dir = '../datasets/Seg_datasets/test/images/'
    masks_dir = '../datasets/Seg_datasets/test/masks/'
    out_fld = './logs/Mask_W/'
    create_mask_array(image_dir, masks_dir, models_weights, out_fld)

    # create masks for test set by using models, without Mobile SAM as agent.
    models_weights = sorted(glob.glob('./logs/Origin1/*pth'))
    image_dir = '../datasets/Origin_Seg_datasets/test/images/'
    masks_dir = '../datasets/Origin_Seg_datasets/test/masks/'
    out_fld = './logs/Mask_W_O/'
    create_mask_array(image_dir, masks_dir, models_weights, out_fld)


def get_ground_predict_severity(true_masks_fld, models_list, predict_masks_fld):
    results = defaultdict(list)
    for m_name in models_list:
        predict_masks = sorted(glob.glob(predict_masks_fld + m_name + '/*'))
        for tfn, pfn in zip(true_masks_fld, predict_masks):
            t_arr = np.load(tfn)
            p_arr = np.load(pfn)
            t1 = (t_arr == 2).sum()
            t2 = (t_arr == 1).sum()
            truth = t1 / (t1 + t2)
            p1 = (p_arr[0, :, :]).sum()
            p2 = (p_arr[1, :, :]).sum()
            predict = p2 / (p1 + p2)
            results['Models'].append(m_name)
            results['Ground truth'].append(truth)
            results['Predict'].append(predict)
    df = pd.DataFrame.from_dict(results)
    return df


def mask_arr2severity():
    # mask arrays converted to severity csv: without Mobile SAM as agent
    true_masks_fld = sorted(glob.glob('../datasets/Seg_datasets/test/masks/*'))
    predict_masks_fld = './logs/Mask_W_O/'
    models_list = os.listdir(predict_masks_fld)
    df = get_ground_predict_severity(true_masks_fld, models_list, predict_masks_fld)
    df.to_csv('./logs/ground_predict_wosam.csv')
    # mask arrays converted to severity csv: with Mobile SAM as agent
    predict_masks_fld = './logs/Mask_W/'
    models_list = os.listdir(predict_masks_fld)
    df = get_ground_predict_severity(true_masks_fld, models_list, predict_masks_fld)
    df.to_csv('./logs/ground_predict_wsam.csv')


if __name__ == "__main__":
    # create_masks_with_mobilesam_or_without()
    mask_arr2severity()