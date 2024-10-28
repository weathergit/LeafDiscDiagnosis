# encoding: utf-8
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, confusion_matrix
from PIL import Image
import seaborn as sns
import matplotlib.ticker as tck
from matplotlib.colors import ListedColormap


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

def make_table2():
    origin_fns_valid = sorted(glob.glob('./logs/Origin1/*logs*csv'))
    origin_fns_test = sorted(glob.glob('./logs/Origin1/*test*csv'))
    sam_fns_valid = sorted(glob.glob('./logs/weights3/*logs*'))
    sam_fns_test = sorted(glob.glob('./logs/weights3/*test*csv'))

    def IoU_table(origin_fns_valid, origin_fns_test, sam_fns_valid, sam_fns_test):
        results = {}
        for f1, f2, f3, f4 in zip(origin_fns_valid, origin_fns_test, sam_fns_valid, sam_fns_test):
            model_name = os.path.basename(f3).split('_')[0]
            o_v_score = pd.read_csv(f1, index_col=0)['valid_score'].max()
            o_t_score = pd.read_csv(f2)['iou_score'].values[0]
            w_v_score = pd.read_csv(f3, index_col=0)['valid_score'].max()
            w_t_score = pd.read_csv(f4)['iou_score'].values[0]
            results[model_name] = [o_v_score, o_t_score, w_v_score, w_t_score]
        res_df = pd.DataFrame.from_dict(results).T
        res_df = res_df.reset_index()
        res_df.columns = ['Models', 'valid_w/o', 'test_w/o', 'valid_w', 'test_w']
        return res_df

    res_df = IoU_table(origin_fns_valid, origin_fns_test, sam_fns_valid, sam_fns_test)
    print(res_df)
    res_df.to_csv('./logs/w_o_sam_resluts.csv')


def make_figure5():
    image_list = ["16-1-1-1(14)_7.jpg", "16-1-1-1(30)_17.jpg",
                  "IMG_0049_3.jpg","IMG_8642_16.jpg",  "IMG_9113_2.jpg", "IMG_9382_6.jpg"]

    image_list_not_stable = ["IMG_2077_3.jpg", "IMG_2102_2.jpg", "IMG_2102_8.jpg",
                             "IMG_2123_14.jpg", "IMG_2198_7.jpg", "IMG_9387_10.jpg"]
    m_names = ['Origin', 'Ground Truth', 'MAENet', 'DeepLab V3+', 'UNet', 'UNet++',
               'DeepLab V3', 'FPN', 'PAN', 'LinkNet']
    model_dir = './logs/with_ms_masks/'
    models_names = os.listdir(model_dir)
    all_imgs_list = []
    for i, name in enumerate(image_list_not_stable):
        origin_fn = '../datasets/Seg_dataset/test/images/' + name
        groud = './logs/ground_mask/' + name
        models_fns = []
        for model in models_names:
            img_fn = model_dir + model + '/' + name
            models_fns.append(img_fn)
        imgs_list = [origin_fn, groud] + models_fns
        all_imgs_list.append(imgs_list)
    fig, axs = plt.subplots(ncols=10, nrows=6, figsize=(6, 4), dpi=200)
    for i, fn_list in enumerate(all_imgs_list):
        for j in range(10):
            img = Image.open(fn_list[j])
            if j == 0:
                img = img.resize((224, 224))
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            if i == 0:
                axs[i, j].set_title(m_names[j], fontsize=7, pad=0.1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2, wspace=0.1, hspace=0)
    plt.savefig('../figs/FigS5.jpg', dpi=500, bbox_inches='tight')
    plt.show()


def make_figures67():
    df = pd.read_csv('./logs/with.csv', index_col=0)
    replace_name = {'maenet': 'MANet', "deeplab3+": "DeepLab V3+", "unet": "UNet", "unetpp": "UNet++",
                    "deeplab3": "DeepLab V3", "fpn": "FPN", "pan": "PAN", "linknet": "LinkNet"}

    df['Models'] = df['Models'].map(replace_name)

    df = df.sort_values(by='Models')
    # ----------------------------Figure 6----------------------------------------
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(6, 3),
                            sharex=True, sharey=True, dpi=300)
    axs = axs.flatten()
    i = 0
    for mm in df['Models'].unique():
        temp = df[df['Models'] == mm]
        axs[i].scatter(temp['Ground truth'], temp["Predict"], s=0.5, color='r', marker="+")
        axs[i].plot([0, 0.7], [0, 0.7], color='k', ls='--', lw=0.8)

        r2 = r2_score(temp['Ground truth'], temp['Predict'])
        r2 = round(r2, ndigits=2)
        axs[i].set_xticks(ticks=np.arange(0.1, 0.9, 0.2),
                          labels=["{0:.1f}".format(i) for i in np.arange(0.1, 0.9, 0.2,)])
        axs[i].set_yticks(ticks=np.arange(0.1, 0.9, 0.2),
                          labels=["{0:.1f}".format(i) for i in np.arange(0.1, 0.9, 0.2)])
        axs[i].set_title(mm, fontsize=9, pad=0)
        axs[i].text(0.08, 0.6, s="$R^2={0}$".format(r2), fontsize=8)

        axs[i].xaxis.set_minor_locator(tck.AutoMinorLocator(2))
        axs[i].yaxis.set_minor_locator(tck.AutoMinorLocator(2))

        # axs[i].xaxis.set_tick_params(which='minor', bottom=False)
        # axs[i].yaxis.set_tick_params(which='minor', bottom=False)

        if i > 3:
            axs[i].set_xlabel('Ground truth', fontsize=9)
        i += 1
    axs[0].set_ylabel('Predicted severity', fontsize=9)
    axs[4].set_ylabel('Predicted severity', fontsize=9)
    plt.savefig('../figs/severity11.jpg', bbox_inches='tight', dpi=300)
    plt.close()
    # plt.show()
    # ----------------------------Figure 7----------------------------------------
    bins = [-0.5, 0.05, 0.25, 0.50, 0.75]
    labels = [1, 3, 5, 7]
    fig, axs = plt.subplots(2, 4, figsize=(6, 3),
                            sharex=True, sharey=True, dpi=300)
    axs = axs.flatten()
    # cmap = ListedColormap(['turquoise', 'brown'])
    i = 0
    for mm in df.Models.unique():
        temp = df[df['Models'] == mm]
        x = temp['Ground truth']
        y = temp['Predict']
        xb = pd.cut(x, bins=bins, labels=labels)
        yb = pd.cut(y, bins=bins, labels=labels)
        matrix = confusion_matrix(xb, yb, normalize='true')
        matrix = matrix.round(decimals=3) # 'RdYlGn_r'
        p = sns.heatmap(matrix, annot=True, ax=axs[i], cbar=False, cmap='RdYlGn_r',
                        annot_kws={'fontsize': 8})
        p.set_xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=[1, 3, 5, 7], fontsize=10)

        p.set_yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=[1, 3, 5, 7], fontsize=10)
        p.set_title(mm, fontsize=9, pad=0)
        if i > 3:
            axs[i].set_xlabel('Predicted', fontsize=9)
        i += 1
    axs[0].set_ylabel('Ground truth', fontsize=9)
    axs[4].set_ylabel('Ground truth', fontsize=9)
    plt.savefig('../figs/headmap_level.jpg', dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # make_table2()
    make_figure5()