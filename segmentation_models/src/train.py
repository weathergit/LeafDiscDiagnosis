# encoding:utf-8

import argparse
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import segmentation_models_pytorch.utils as segu
import torch
from dataloader.dataload import load_data
from models.MODEL_LIST import model_list
from losses.common_loss import loss
from metrics.common_metrics import metrics


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(project, train_loader, valid_loader, model, loss, metrics, epochs=300, patience=10):
    seed_everything()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    train_epoch = segu.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer,
                                        device=device, verbose=True)

    valid_epoch = segu.train.ValidEpoch(model, loss=loss, metrics=metrics, device=device, verbose=True)

    max_score = 0.0

    logs = defaultdict(list)

    counter = 0
    for i in range(epochs):
        print('\n Epoch:{}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        logs['train_loss'].append(train_logs['dice_loss'])
        logs['valid_loss'].append(valid_logs['dice_loss'])

        logs['train_score'].append(train_logs['iou_score'])
        logs['valid_score'].append(valid_logs['iou_score'])

        logs['train_fscore'].append(train_logs['fscore'])
        logs['valid_fscore'].append(valid_logs['fscore'])

        if max_score < valid_logs['iou_score']:
            counter = 0
            max_score = valid_logs['iou_score']
            best_weights = project + '_best.pth'
            torch.save(model, './logs/' + best_weights)
            print('Model saved!')
        else:
            counter += 1
            print('Counter {} of 10'.format(counter))

        if counter == patience:
            print('Early stopping with best score {:4f}'.format(max_score))
            break

    df = pd.DataFrame.from_dict(logs)
    df.to_csv('./logs/' + project + '_logs.csv', encoding='utf-8')
    torch.cuda.empty_cache()


def main(args):
    train_loader, valid_loader = load_data(data_dir=args.data_dir)
    model = model_list[args.model_name]
    train(project=args.model_name, model=model, train_loader=train_loader,
          valid_loader=valid_loader, loss=loss, metrics=metrics)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    main(args)
