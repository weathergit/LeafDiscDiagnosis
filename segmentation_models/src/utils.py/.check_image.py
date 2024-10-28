import glob
import shutil

from seg_models import ENCODER, ENCODER_WEIGHTS, CLASSES
import os
import segmentation_models_pytorch as seg
from DownyData import DownyData
from src.dataloader.data_aug import get_val_aug, get_processing

def check_shape():
    DATA_dir = '../seg_datasets/'
    img_train_dir = os.path.join(DATA_dir, 'train/images/')
    mask_train_dir = os.path.join(DATA_dir, 'train/masks/')

    img_val_dir = os.path.join(DATA_dir, 'val/images/')
    mask_val_dir = os.path.join(DATA_dir, 'val/masks/')

    img_test_dir = os.path.join(DATA_dir, 'test/images/')
    mask_test_dir = os.path.join(DATA_dir, 'test/masks/')

    preprocess_fn = seg.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # train_dataset = DownyData(images_dir=img_train_dir, masks_dir=mask_train_dir, augmentation=get_train_aug(),
    #                           preprocessing=get_processing(preprocess_fn), classes=CLASSES)
    #
    # valid_dataset = DownyData(images_dir=img_val_dir, masks_dir=mask_val_dir, augmentation=get_val_aug(),
    #                           preprocessing=get_processing(preprocess_fn), classes=CLASSES)

    test_dataset = DownyData(images_dir=img_test_dir, masks_dir=mask_test_dir, augmentation=get_val_aug(),
                              preprocessing=get_processing(preprocess_fn), classes=CLASSES)
    # img = Image.open('../seg_datasets/train/images/IMG_9322.JPG')
    # arr = np.load('../seg_datasets/train/masks/IMG_9322.npy')
    #
    # fig, axs = plt.subplots(1,2, dpi=400)
    # axs[0].imshow(np.asarray(img))
    # axs[1].imshow(arr)
    # plt.show()

    # for img, msk in train_dataset:
    #     print(img.shape, msk.shape)

    for img, msk in test_dataset:
        print(img.shape, msk.shape)


def move_wrong_boxes():
    fns = glob.glob('../crop_datasets/wrong_clip/wrong_boxes/*')
    for fn in fns:
        basename = os.path.basename(fn)[:-4]
        src = '../crop_datasets/crop_masks/' + basename +'.npy'
        dst = '../crop_datasets/wrong_clip/wrong_mask'
        shutil.move(src, dst)


if __name__ == '__main__':
    move_wrong_boxes()
