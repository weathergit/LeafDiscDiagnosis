# encoding:utf-8

import os
import glob
import pandas as pd
import torch
from dataloader.DownyData import DownyData
from dataloader.data_aug import get_processing, get_val_aug
from models.base_model import ENCODER, ENCODER_WEIGHTS,CLASSES
from models.base_model import model_list
import segmentation_models_pytorch as seg
from torch.utils.data import DataLoader
import segmentation_models_pytorch.utils as segu
import argparse
import time


def main(args):
    start = time.time()
    DATA_dir = args.data_dir
    img_test_dir = os.path.join(DATA_dir, 'test/images/')
    mask_test_dir = os.path.join(DATA_dir, 'test/masks/')
    preprocess_fn = seg.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    test_datasets = DownyData(images_dir=img_test_dir, masks_dir=mask_test_dir, augmentation=get_val_aug(),
                              preprocessing=get_processing(preprocess_fn), classes=CLASSES)

    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=8)

    loss = segu.losses.DiceLoss()
    metrics = [segu.metrics.IoU(threshold=0.5)]

    weights_dir = glob.glob(args.weights_dir)
    model_weights = {w.split('/')[-1][:-9]: w for w in weights_dir}

    weight = model_weights[args.model_name]

    model = torch.load(weight)

    test_epoch = segu.train.ValidEpoch(model=model, loss=loss, metrics=metrics, device='cuda')

    logs = test_epoch.run(test_loader)

    torch.cuda.empty_cache()
    end = time.time()
    duration = end - start
    frame_per_second = 532 / duration
    print('FPS of {0} is {1:.2f}'.format(args.model_name, frame_per_second))

    df = pd.DataFrame(logs, index=[args.model_name])
    df.to_csv('./logs/'+args.model_name+'_test.csv', encoding='utf-8')


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument("--weights_dir", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    main(args)