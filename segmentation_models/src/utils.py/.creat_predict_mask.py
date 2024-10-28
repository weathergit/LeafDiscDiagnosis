import os
import sys

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SRC_PATH + '/..')
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks

import segmentation_models_pytorch as seg

from models.base_model import ENCODER, CLASSES, ENCODER_WEIGHTS
from dataloader.DownyData import DownyData
from dataloader.data_aug import get_val_aug, get_processing

plt.rcParams["savefig.bbox"] = 'tight'
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def show(imgs, outname):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        img = img.resize((224, 224))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(outname)


def get_img_list(DATA_dir, tvt='test'):
    img_test_dir = os.path.join(DATA_dir, tvt + '/images/')
    mask_test_dir = os.path.join(DATA_dir, tvt + '/masks/')
    preprocess_fn = seg.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    test_dadasets = DownyData(images_dir=img_test_dir, masks_dir=mask_test_dir, augmentation=get_val_aug(),
                              preprocessing=get_processing(preprocess_fn), classes=CLASSES)
    data_len = len(test_dadasets)
    test_loader = DataLoader(test_dadasets, batch_size=1, shuffle=False, num_workers=8)
    image_names = test_dadasets.ids
    assert len(image_names) == data_len
    print('datasets load')
    return test_loader, image_names


def get_models_list():
    weights = glob.glob('../logs/weights/*pth')
    models_list = {}
    for weight in weights:
        name = os.path.basename(weight)[:-4]
        best_model = torch.load(weight, map_location='cuda')
        models_list[name] = best_model.eval()
    print('models load')
    return models_list


def main():
    data_dir = '/home/qtian/Documents/Grape_Severity_Estimation/UNet_segment/datasets/Seg_datasets/'
    task = 'train'
    img_loader, img_names = get_img_list(DATA_dir=data_dir, tvt=task)
    models_list = get_models_list()
    for k, model in models_list.items():
        for batch, name in zip(img_loader, img_names):
            print(k, name)
            img, mask = batch
            pr_mask = model.predict(img.to(device))
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            outdir = '../logs/masks/parrs/' + k + '/'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            np.save(outdir + name[:-4], pr_mask)
        torch.cuda.empty_cache()


def create_origin_mask():
    original_images = sorted(glob.glob('../../datasets/Seg_datasets/*/images/*'))
    original_masks = sorted(glob.glob('../../datasets/Seg_datasets/*/masks/*'))
    assert len(original_images) == len(original_masks)
    for ipath, mpath in zip(original_images, original_masks):
        bname1 = os.path.basename(ipath)[:-4]
        bname2 = os.path.basename(mpath)[:-4]
        assert bname1 == bname2
        img = read_image(ipath)
        ground = np.load(mpath)
        ground_list = [(ground == v) for v in [1, 2]]
        ground = np.stack(ground_list, axis=0).astype('float')
        ground = torch.from_numpy(ground).bool()
        with_masks = [draw_segmentation_masks(img, masks=ground, alpha=0.3, colors=['green', 'red'])]
        outname = '../logs/masks/origin/' + bname1 + '.jpg'
        show(with_masks, outname)


def create_predict_mask():
    original_images = sorted(glob.glob('../../datasets/Seg_datasets/*/images/*'), key=lambda x: os.path.basename(x))
    model_names = os.listdir('../logs/masks/parrs/')
    for mname in model_names:
        pattern = '../logs/masks/parrs/' + mname + '/*'
        arr_list = sorted(glob.glob(pattern))
        assert len(original_images) == len(arr_list)
        for ipath, ppath in zip(original_images, arr_list):
            bname1 = os.path.basename(ipath)[:-4]
            bname2 = os.path.basename(ppath)[:-4]
            assert bname1 == bname2
            print(mname, bname2)
            img = read_image(ipath)
            img = v2.Resize(size=(224, 224), antialias=True)(img)
            mask = np.load(ppath)
            mask = torch.from_numpy(mask).bool()
            with_masks = [draw_segmentation_masks(img, masks=mask, alpha=0.3, colors=['green', 'red'])]
            outdir = '../logs/masks/plabels/' + mname + '/'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outname = outdir + bname1 + '.jpg'
            show(with_masks, outname)


if __name__ == '__main__':
    create_predict_mask()
