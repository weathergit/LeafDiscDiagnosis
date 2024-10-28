# encoding:utf-8

import glob
import os
import subprocess
import torch


def main():
    model_names = [os.path.basename(fn) for fn in
                   glob.glob('./logs/weights/yolov8?.pt')]
    # print(model_names)

    for name in model_names:
        command = ['python', 'yolo_train.py', '--type', name]
        subprocess.run(command)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()