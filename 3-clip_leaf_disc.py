# encoding : utf-8
import os
import glob
import numpy as np
import pandas as pd 
import json
import cv2
import shutil
import matplotlib.pyplot as plt 


def clip_leaf_disc(msk_arr_path, img_path, json_path, out_img_dir, out_msk_dir):
    """crop the images and masks using new_label_xyxy (xyxy).
    datasets in new_datasets_mask.
    Args:
        msk_arr_path (str):path of mask array 
        img_path (str): path of image
        json_path (str):path of json
        out_img_dir (str): output clipped image
        out_msk_dir (str): output clipped mask
    """
    mask_arr = np.load(msk_arr_path)
    image = cv2.imread(img_path)
    basename = os.path.basename(img_path)[:-4]
    with open(json_path, 'rb') as f:
        masks = json.load(f)['shapes']
    df = pd.DataFrame(masks)
    df = df.query("label == 'disc'")
    polygons = df['points'].to_list()
    for i, polygon in enumerate(polygons):
        
        template1 = np.zeros_like(image)
        template2 = np.zeros_like(mask_arr)
        
        poly = np.asarray(polygon, dtype=np.int32)
        cv2.fillPoly(template1, [poly], (255,255,255))
        cv2.fillPoly(template2, [poly], (255,255,255))
        
        x, y, w, h = cv2.boundingRect(poly)
        # ****** important code *******
        # if clip original images and ams, comment the below code
        # img_result = cv2.bitwise_and(image, template1)
        msk_result = cv2.bitwise_and(mask_arr, template2)

        # clip original images
        img_result = image

        leaf_disc = img_result[y:y+h, x:x+w]
        leaf_disc_mask = msk_result[y:y+h, x:x+w]
        
        outname1 = out_img_dir + basename + f'_{i+1}.jpg'
        outname2 = out_msk_dir + basename + f'_{i+1}'
        
        cv2.imwrite(outname1, leaf_disc)
        np.save(outname2, np.asarray(leaf_disc_mask))


def clip_with_background_remove():
    """
    这段代码是为了提取出每一个叶盘. polygon是标注的每一个叶盘的范围.
    """
    polys = sorted(glob.glob('./new_datasets/2.leaf_disc/*.json'))
    images = sorted(glob.glob('./new_datasets/2.leaf_disc/*.JPG'))
    label_arrays = sorted(glob.glob('./new_datasets/3.3new_datasets_mask/SegmentationClass/*npy')) 

    out_img_dir = './new_datasets/5.3clipped_origin/images/'
    out_msk_dir = './new_datasets/5.3clipped_origin/masks/'
    os.mkdir(out_img_dir)
    os.mkdir(out_msk_dir)
    for poly_fn, image_fn, lar_fn in zip(polys, images, label_arrays):
        b1 = os.path.basename(poly_fn)[:-5]
        b2 = os.path.basename(image_fn)[:-4]
        b3 = os.path.basename(lar_fn)[:-4]
        assert b1==b2==b3
        print(b2)
        clip_leaf_disc(lar_fn, image_fn, poly_fn, out_img_dir, out_msk_dir)



def get_selected_fn_name():
    """get the slelected leaf disc image's file name
    """
    selected_fns = './new_datasets/5.2.clipped_datasets_old/select_images/*'
    fns = glob.glob(selected_fns)
    basenames = []
    for fn in fns:
        basenames.append(os.path.basename(fn))
    df = pd.DataFrame.from_dict({'fn': basenames})
    df.to_csv('./src/selected_fn.csv',index=None)


def move_selected_image_masks():
    """copy selected images to new folders
    """
    df = pd.read_csv('./src/selected_fn.csv')
    for fn in df['fn']:
        img_src = './new_datasets/5.3clipped_origin/images/' + fn
        mask_src = './new_datasets/5.3clipped_origin/masks/' + fn[:-4] +'.npy'
        img_dst = './new_datasets/5.3clipped_origin/select_images/' + fn
        mask_dst = './new_datasets/5.3clipped_origin/select_masks/' + fn[:-4] + '.npy'
        shutil.copy(img_src, img_dst)
        shutil.copy(mask_src, mask_dst)


def clip_without_mask():
    images = sorted(glob.glob('./new_datasets/new_datasets_mask/JPEGImages/*'))
    bounds = sorted(glob.glob('./new_datasets/new_label_xyxy/*'))
    mask_arrs = sorted(glob.glob('./new_datasets/new_datasets_mask/SegmentationClass/*'))
    
    print(len(images), len(bounds), len(mask_arrs))
    assert len(images) == len(bounds) == len(mask_arrs)

    for f1, f2,f3 in zip(images, bounds, mask_arrs):
        image = cv2.imread(f1)
        mask = np.load(f3)
        print(f1)
        with open(f2, 'r') as f:
            i = 0 
            for line in f.readlines():
                _, x1, y1, x2, y2 = line.split(' ')
                x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
                new_img = image[y1:y2,x1:x2]
                new_msk = mask[y1:y2, x1:x2]
                outname1 = './new_datasets/Origin_Crop/images/' + os.path.basename(f1)[:-4] + f"_{i+1}"
                outname2 = './new_datasets/Origin_Crop/masks/' + os.path.basename(f1)[:-4] + f"_{i+1}"
                cv2.imwrite(outname1+'.jpg',new_img)
                np.save(outname2, new_msk)
                i += 1


def old_code():
    arr = np.load(lar)
    img = cv2.imread(image)
    with open(poly,'rb') as f:
        masks =  json.load(f)['shapes']

    df = pd.DataFrame(masks)
    df1 = df[df['label']=='disc']
    
    polygons = df1['points'].tolist()

    templates = np.zeros_like(img)
    
    for polygon in polygons:
        # here polygon must be in np.int32(other dtypes), a list of points is uncorrect.
        polygon = np.asarray(polygon, dtype=np.int32)
        cv2.fillPoly(templates, [polygon], (255, 255, 255))

    result = cv2.bitwise_and(img, templates)
    
    img_outdir = './new_datasets/new_datasets_clip/images_white/'
    mask_outdir = './new_datasets/new_datasets_clip/masks_white/'
    
    for i, polygon in enumerate(polygons):
        polygon = np.asarray(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(polygon)
        # here reports a bug, if the clipped leaf disc images contains part of other leaf disc, \
        # the mask would be not correct. 
        roi = result[y:y + h, x:x + w]
        arr_roi = arr[y:y + h, x:x + w]
        outname1 = img_outdir + b1 + f'_{i+1}.jpg'
        outname2 = mask_outdir + b1 + f'_{i+1}'
        cv2.imwrite(outname1, roi)
        np.save(outname2, arr_roi)


if __name__ == '__main__':
    move_selected_image_masks()
    # clip_with_background_remove()