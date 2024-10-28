import  os
import shutil
import glob
import numpy as np

def split():
    """
    split leaf disc images from model training.
    manual create folders train/val/test   images/masks
    """

    dst_folder = './new_datasets/6.Origin_dataset/'

    images = sorted(glob.glob('./new_datasets/5.2clipped_origin/select_images/*'))
    masks = sorted(glob.glob('./new_datasets/5.2clipped_origin/select_masks/*'))
    assert  len(images) == len(masks)
    data_number = len(images)
    k1 = int(data_number*0.6)
    k2 = int(data_number*0.2)
    
    # 20%
    k3 = data_number - k1 - k2

    k1_2 = k1 + k2

    np.random.seed(42)
    np.random.shuffle(images)
    np.random.seed(42)
    np.random.shuffle(masks)

    for i, (ipth, mpth) in enumerate(zip(images,masks)):
        basename_i = os.path.basename(ipth)
        basename_m = os.path.basename(mpth)
        if i <= k1: 
            img_dst = dst_folder+'train/images/'+basename_i
            mask_dst = dst_folder +'train/masks/'+basename_m
            
            shutil.copy(ipth,dst=img_dst)
            shutil.copy(mpth,dst=mask_dst)
        elif i > k1 and i <= k1_2:
            img_dst = dst_folder+'val/images/'+basename_i
            mask_dst = dst_folder +'val/masks/'+basename_m
            shutil.copy(ipth,dst=img_dst)
            shutil.copy(mpth,dst=mask_dst)
        elif i > k1_2:
            img_dst = dst_folder+'test/images/' + basename_i
            mask_dst = dst_folder +'test/masks/' + basename_m
            shutil.copy(ipth,dst=img_dst)
            shutil.copy(mpth,dst=mask_dst)


if __name__ == '__main__':
    split()



