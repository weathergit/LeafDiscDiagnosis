# encoding: utf-8
import subprocess
from models.MODEL_LIST import model_list


def main():
    data_dir = '../datasets/Seg_dataset/'
    weights_dir = './logs/with_ms_logs/*pth'
    for name in model_list.keys():
        print('{} is testing'.format(name))
        command = ['python', 'test.py', '--model_name', name,
                   '--data_dir', data_dir, '--weights_dir',
                   weights_dir]
        subprocess.run(command)
        print('{} is done'.format(name))


if __name__ == "__main__":
    main()