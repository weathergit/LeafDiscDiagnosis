import sys
import glob
from ultralytics import YOLO
import torch
from ultralytics import settings
import pandas as pd
# import argparse

settings.update({
    'datasets_dir': "/home/qtian/Documents/Grape_Severity_Estimation/YOLO_detect/datasets/"
}
)
# this script will print the FPS: Frame per second

def test_multiple_yolo():
    file_log = open('./logs/logs.txt', 'w')
    sys.stdout = file_log
    model_weights = sorted(glob.glob('./logs/yolo*/train/weights/best.pt'))
    model_cfgs = sorted(glob.glob('../configs/y*'))
    projects = [fn.split('/')[-1][:-5] for fn in model_cfgs]
    results = {}
    for project, weight, cfg in zip(projects, model_weights, model_cfgs):
        model = YOLO(cfg).load(weight)
        project_name = './logs/'+project+'/test/'
        metrics = model.val(data='./configs/mydata.yaml',
                            split='test', project=project_name,
                            device='cuda')
        results[project] = metrics.box.map50
        torch.cuda.empty_cache()
    print(results)
    file_log.close()
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('./logs/test_map.csv')

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--type', default=None, type=str)
#     args = parser.parse_args()
#     return args


if __name__ == '__main__':
    # args = main()
    test_multiple_yolo()
