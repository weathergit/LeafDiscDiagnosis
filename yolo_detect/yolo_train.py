# encoding:utf-8
import os.path

from ultralytics import YOLO
from ultralytics import settings
import argparse
# from pprint import pprint

settings.update({
    'datasets_dir': "/home/qtian/Documents/Grape_Severity_Estimation/YOLO_detect/datasets/",
    'runs_dir': '/home/qtian/Documents/Grape_Severity_Estimation/YOLO_detect/src/logs/',
    'weights_dir': '/home/qtian/Documents/Grape_Severity_Estimation/YOLO_detect/src/logs/weights/',
})


def train_yolo(args):
    weight = args.type
    model_cfg = './configs/'+args.type[:-3]+'.yaml'
    project_name = ('/home/qtian/Documents/Grape_Severity_Estimation/YOLO_detect/src/logs/'
                    + args.type[:-3])
    # print(model_cfg)
    model = YOLO(model_cfg).load(weight)
    model.train(data='./configs/mydata.yaml', epochs=150, patience=10,
                seed=42, project=project_name)
    model.val()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = main()
    train_yolo(args)