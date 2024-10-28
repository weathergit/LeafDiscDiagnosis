import glob
import os
from collections import defaultdict
import numpy as np
import pandas as pd


def get_ground_predict_severity(true_masks_fld, models_list, predict_masks_fld):
    results = defaultdict(list)
    for m_name in models_list:
        predict_masks = sorted(glob.glob(predict_masks_fld+ m_name +'/*.npy'))
        for tfn, pfn in zip(true_masks_fld, predict_masks):
            t_arr = np.load(tfn)
            p_arr = np.load(pfn)
            basename = os.path.basename(tfn)
            t1 = (t_arr == 2).sum()
            t2 = (t_arr == 1).sum()
            truth = t1 / (t1 + t2)
            p1 = (p_arr[0,:,:]).sum()
            p2 = (p_arr[1,:,:]).sum()
            predict = p2 / (p1 + p2)
            results['Path'].append(basename)
            results['Models'].append(m_name)

            results['t_leaf'].append(t2)
            results['t_downy'].append(t1)
            results['Ground truth'].append(truth)

            results['p_leaf'].append(p1)
            results['p_downy'].append(p2)
            results['Predict'].append(predict)
    df = pd.DataFrame.from_dict(results)
    return df


def severity2df():
    true_masks_fld = sorted(glob.glob('../datasets/Seg_dataset/test/masks/*'))
    predict_masks_fld = './logs/with_ms_masks/'
    # true_masks_fld = sorted(glob.glob('../datasets/Origin_dataset/test/masks/*'))
    # predict_masks_fld = './logs/without_ms_masks/'
    models_list = os.listdir(predict_masks_fld)
    df = get_ground_predict_severity(true_masks_fld, models_list, predict_masks_fld)
    df.to_csv('./logs/with.csv')
    # df.to_csv('./logs/without.csv')


if __name__ == "__main__":
    severity2df()