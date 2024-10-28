import  os
import shutil
import glob
import numpy as np

def split():
    """
    Split original dataset into train, val ,test for YOLO models.
    """
    dst_folder = './new_datasets/4.YOLO_datasets/'

    images = sorted(glob.glob('./new_datasets/3.3new_datasets_mask/JPEGImages/*'))
    labels = sorted(glob.glob('./new_datasets/3.1new_label_xywh/*'))
    assert  len(images) == len(labels)
    
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

    for i, (ipth, mpth) in enumerate(zip(images,labels)):
        basename_i = os.path.basename(ipth)
        basename_m = os.path.basename(mpth)
        if i <= k1: 
            img_dst = dst_folder+'images/train/'+basename_i
            label_dst= dst_folder +'labels/train/'+basename_m
            shutil.copy(ipth,dst=img_dst)
            shutil.copy(mpth,dst=label_dst)
        elif i > k1 and i <= k1_2:
            img_dst = dst_folder+'images/train/'+basename_i
            label_dst= dst_folder +'labels/train/'+basename_m
            shutil.copy(ipth,dst=img_dst)
            shutil.copy(mpth,dst=label_dst)
        elif i > k1_2:
            img_dst = dst_folder+'images/train/'+basename_i
            label_dst= dst_folder +'labels/train/'+basename_m
            shutil.copy(ipth,dst=img_dst)
            shutil.copy(mpth,dst=label_dst)

if __name__ == '__main__':
    split()



