# encoding:utf-8

import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataloader.dataload import load_data
from models.UNet_Class import UCNet
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU, Fscore


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train():
    seed_everything()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data_dir = '../datasets/Seg_datasets/'
    train_loader, valid_loader = load_data(data_dir=data_dir)

    model = UCNet()
    model.to(device)

    criterion1 = DiceLoss(mode='multilabel')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    best_miou = 0
    best_accuracy = 0

    epochs = 300

    save_dict = defaultdict(list)

    for epoch in range(epochs):
        train_loss = 0.0
        # train_loss1 = 0.0
        # train_loss2 = 0.0
        train_iou = 0.0
        train_fscore = 0.0
        # train_accuracy = 0.0

        model.train()
        train_bar = tqdm(train_loader, total=len(train_loader), colour='green')
        for data in train_bar:
            image, mask = data
            y1 = model(image.to(device))
            loss = criterion1(y1, mask.to(device))
            # cp = criterion2(y2, severity.to(device))
            # loss = (dice + cp) / 2

            iou = IoU(threshold=0.5)(y1, mask.to(device))
            fscore = Fscore()(y1, mask.to(device))

            # _, y2_pred = torch.max(y2, dim=1)
            # acc = (y2_pred == severity.to(device)).sum() / 64

            loss.backward()
            optimizer.step()

            # train_loss1 += dice.item()
            # train_loss2 += cp.item()
            train_loss += loss.item()
            train_iou += iou.item()
            train_fscore += fscore.item()
            # train_accuracy += acc.item()

            train_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            train_bar.set_postfix(loss=loss.item(), iou=iou.item(), fscore=fscore.item())

        mean_train_loss = train_loss / (len(train_loader))
        # mean_train_loss1 = train_loss1 / (len(train_loader))
        # mean_train_loss2 = train_loss2 / (len(train_loader))
        mean_train_iou = train_iou / (len(train_loader))
        mean_train_fscore = train_fscore / (len(train_loader))
        # mean_train_accuracy = train_accuracy / (len(train_loader))

        val_loss = 0.0
        # val_loss1 = 0.0
        # val_loss2 = 0.0
        val_iou = 0.0
        val_fscore = 0.0
        # val_accuracy = 0.0

        model.eval()
        with torch.no_grad():
            val_bar = tqdm(valid_loader, total=len(valid_loader), colour='blue')
            for data in val_bar:
                image, mask = data
                y1 = model(image.to(device))
                loss = criterion1(y1, mask.to(device))
                # cp = criterion2(y2, severity.to(device))
                # loss = (dice + cp) / 2

                iou = IoU(threshold=0.5)(y1, mask.to(device))
                fscore = Fscore()(y1, mask.to(device))

                # _, y2_pred = torch.max(y2, dim=1)
                # acc = (y2_pred == severity.to(device)).sum()

                val_loss += loss.item()
                # val_loss1 += dice.item()
                # val_loss2 += cp.item()
                val_iou += iou.item()
                val_fscore += fscore.item()
                # val_accuracy += acc.item()

                val_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                val_bar.set_postfix(loss=loss.item(), iou=iou.item(), fscore=fscore.item())

            mean_val_loss = val_loss / len(valid_loader)
            # mean_val_loss1 = val_loss1 / len(valid_loader)
            # mean_val_loss2 = val_loss2 / len(valid_loader)
            mean_val_iou = val_iou / len(valid_loader)
            mean_val_fscore = val_fscore / len(valid_loader)
            # mean_val_accuracy = val_accuracy / len(valid_loader)

        print('Train:loss={0} mIoU={1} Fscore={2}'.format(mean_train_loss, mean_train_iou,mean_train_fscore))
        print('Val:loss={0} mIoU={1} Fscore={2}'.format(mean_val_loss, mean_val_iou,mean_val_fscore))

        # save_dict['train_dic2'].append(mean_train_loss1)
        # save_dict['train_cp'].append(mean_train_loss2)
        save_dict['train_loss'].append(mean_train_loss)
        save_dict['train_mIoU'].append(mean_train_iou)
        save_dict['train_fscore'].append(mean_train_fscore)
        # save_dict['train_accuray'].append(mean_train_accuracy)

        # save_dict['val_dic2'].append(mean_val_loss1)
        # save_dict['val_cp'].append(mean_val_loss2)
        save_dict['val_loss'].append(mean_val_loss)
        save_dict['val_mIoU'].append(mean_val_iou)
        save_dict['val_fscore'].append(mean_val_fscore)
        # save_dict['val_accuray'].append(mean_val_accuracy)

        df = pd.DataFrame.from_dict(save_dict)
        df.to_csv('./logs/UCNet.csv', encoding='utf-8')

        if mean_val_iou > best_miou:
            counter = 0
            best_miou = mean_val_iou
            # best_accuracy = mean_val_accuracy
            torch.save(model.state_dict(), './logs/UCNet_best.pth')

    print('model training finished with best mIoU:{:.4f} best accuracy:{:4f}'.format(best_miou, best_accuracy))


if __name__ == "__main__":
    train()
