import numpy as np
import torch.cuda
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns


def print_map50():
    """
    seg
    """
    fns = sorted(glob.glob('../crop_datasets/crop_output/BD_BCE/*log*'))
    for fn in fns:
        basename = os.path.basename(fn)
        model_name = fn.split('/')[3].split('_')[0]
        df = pd.read_csv(fn)
        map50 = df['valid_score']
        print(model_name, round(map50.max(), 4))


def print_yolo_map50():
    fns = sorted(glob.glob('../yolo/new_output/*/train/results.csv'))
    for fn in fns:
        basename = os.path.basename(fn)
        model_name = fn.split('/')[3]
        df = pd.read_csv(fn)
        map50 = df[df.columns[6]]
        print(model_name, round(map50.max(), 4))


def seg_model_size():
    model_weights = sorted(glob.glob('../crop_datasets/crop_output/*pth'))
    for weight in model_weights:
        fn = weight.split('/')[3][:-4]
        model = torch.load(weight)
        total_paras = sum(p.numel() for p in model.parameters())
        print('{}:{}'.format(fn, total_paras / 1e6))


def yolo_model_size():
    model_weights = sorted(glob.glob('../yolo/new_output/*/train/weights/best.pt'))
    for weight in model_weights:
        fn = weight.split('/')[3]
        model = torch.load(weight)
        total_paras = sum(p.numel() for p in model['model'].parameters())
        print('{} model zie {}'.format(fn, total_paras / 1e6))


def compare_masks():
    origin_images = sorted(glob.glob('../crop_datasets/Seg_datasets/test/images/*'))
    ground_truth = sorted(glob.glob('../crop_datasets/Seg_datasets/test/masks/*'))
    model_names = os.listdir('../crop_datasets/Seg_predict_masks/')
    model_pr_dict = {}
    model_pr_dict.update({'origin': origin_images})
    model_pr_dict.update({'ground': ground_truth})
    for name in model_names:
        pattern = '../crop_datasets/Seg_predict_masks/' + name + '/*'
        pr_list = sorted(glob.glob(pattern))
        model_pr_dict[name] = pr_list
    fig, axs = plt.subplots(nrows=1, ncols=10)
    for i in range(len(origin_images)):
        j = 0
        for key, value in model_pr_dict.items():
            if key == 'origin':
                img = Image.open(value[i])
                img = img.resize((224, 224))
                axs[j].imshow(np.asarray(img))
                if i == 0:
                    axs[j].set_title(key, fontsize=6, pad=0)
                axs[j].axis('off')
            elif key == 'ground':
                arr = np.load(value[i])
                if arr.shape[0] == 2:
                    arr0 = arr[0]
                    arr1 = arr[1]
                    x, y = np.where(arr1 == 1)
                    arr0[x, y] = 2
                else:
                    arr0 = arr
                axs[j].imshow(arr0)
                if i == 0:
                    axs[j].set_title(key, fontsize=6, pad=0)
                axs[j].axis('off')
            else:
                arr = np.load(value[i])
                arr0, arr1 = arr[0], arr[1]
                x, y = np.where(arr1 == 1)
                arr0[x, y] = 2
                axs[j].imshow(arr0)
                if i == 0:
                    axs[j].set_title(key, fontsize=6, pad=0)
                axs[j].axis('off')
            j += 1
        plt.subplots_adjust(wspace=0)
        # plt.show()
        plt.savefig('../crop_datasets/mask_compare/' + str(i) + '.jpg',
                    dpi=500, bbox_inches='tight')


def one2oneline_rmse():
    ground_truth = sorted(glob.glob('../crop_datasets/Seg_datasets/test/masks/*'))
    model_names = os.listdir('../crop_datasets/Seg_predict_masks/')
    model_pr_dict = {}
    model_pr_dict.update({'ground': ground_truth})
    for name in model_names:
        pattern = '../crop_datasets/Seg_predict_masks/' + name + '/*'
        pr_list = sorted(glob.glob(pattern))
        model_pr_dict[name] = pr_list
    results = defaultdict(list)
    for i in range(len(ground_truth)):
        for key, value in model_pr_dict.items():
            arr = np.load(value[i])
            if key == 'ground':
                d1 = arr[arr == 2].sum()
                d2 = arr[arr == 1].sum()
                dis = d1 / (d1 + d2)
            else:
                d1 = arr[1].sum()
                d2 = arr[0].sum()
                dis = d1 / (d1 + d2)
            results[key].append(dis)
    df = pd.DataFrame.from_dict(results)
    df.to_csv('../crop_datasets/compare_dsieases_rmse.csv')


def rmse_plot():
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 5),
                            sharex=True, sharey=True)
    axs = axs.flatten()
    df = pd.read_csv('../crop_datasets/compare_dsieases_rmse.csv', index_col=0)
    x = df[df.columns[0]]
    i = 0
    for col in df.columns[1:]:
        y = df[col]
        print(col)
        axs[i].scatter(x, y, s=3, color='r')
        axs[i].plot([0, 0.7], [0, 0.7], color='k', ls='--')
        axs[i].set_xticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                          labels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        axs[i].set_yticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                          labels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        axs[i].set_title(col, fontsize=12, pad=0)
        if i > 3:
            axs[i].set_xlabel('Severity', fontsize=10)
        axs[i].set_ylabel('Predicted severity', fontsize=10)
        i += 1
    # plt.show()
    plt.savefig('../../yolo_seg_Unet/figs/rmse.jpg', dpi=500,
                bbox_inches='tight')


def diseases_bar():
    df = pd.read_csv('../crop_datasets/compare_dsieases_rmse.csv', index_col=0)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.hist(x=df['ground'], color='lightblue', edgecolor='k')
    plt.savefig('../../yolo_seg_Unet/figs/diseases.jpg', dpi=500,
                bbox_inches='tight')
    plt.show()


def plot_confusion_matrix():
    df = pd.read_csv('../crop_datasets/compare_dsieases_rmse.csv', index_col=0)

    bins = [-0.5, 0.05, 0.25, 0.50, 0.75]
    labels = [1, 3, 5, 7]
    x = df['ground']
    xb = pd.cut(x, bins=bins, labels=labels)

    fig, axs = plt.subplots(2, 4, figsize=(10, 6),
                            sharex=True, sharey=True)
    axs = axs.flatten()
    i = 0
    for col in df.columns[1:]:
        y = df[col]
        yb = pd.cut(y, bins=bins, labels=labels)
        matrix = confusion_matrix(xb, yb, normalize='true')
        matrix = matrix.round(decimals=3)
        p = sns.heatmap(matrix, annot=True, ax=axs[i], cbar=False, cmap='RdYlGn_r')
        p.set_xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=[1, 3, 5, 7], fontsize=10)

        p.set_yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=[1, 3, 5, 7], fontsize=10)
        p.set_title(col, fontsize=10)
        i += 1
    # plt.savefig('../../yolo_seg_Unet/figs/confusion.jpg', dpi=500,
    #             bbox_inches='tight')
    # plt.show()


# def threshold_segment_compare():
#     ground_truth = sorted(glob.glob('../crop_datasets/Seg_datasets/test/masks/*'))
#     model_names = os.listdir('../crop_datasets/Seg_predict_masks_origin/')
#     model_pr_dict = {}
#     model_pr_dict.update({'ground': ground_truth})
#     for name in model_names:
#         pattern = '../crop_datasets/Seg_predict_masks_origin/' + name + '/*'
#         pr_list = sorted(glob.glob(pattern))
#         model_pr_dict[name] = pr_list
#     for thresold in np.arange(0.1, 1.0, 0.1):
#         results = defaultdict(list)
#         for i in range(len(ground_truth)):
#             for key, value in model_pr_dict.items():
#                 arr = np.load(value[i])
#                 if key == 'ground':
#                     d1 = arr[arr == 2].sum()
#                     d2 = arr[arr == 1].sum()
#                     dis = d1 / (d1 + d2)
#                     # if new diseases index
#                     # dis = np.sqrt(d1+d2) * dis
#                 else:
#                     d1 = (arr[1] > thresold).sum()
#                     d2 = arr[0].sum()
#                     dis = d1 / (d1 + d2)
#                     # dis = np.sqrt(d1 + d2) * dis
#                 results[key].append(dis)
#         df = pd.DataFrame.from_dict(results)
#         df.to_csv('../crop_datasets/dsieases_rmse_'
#                   +str(round(thresold,2))+'.csv')


if __name__ == '__main__':
    # one2oneline_rmse()
    print_map50()
