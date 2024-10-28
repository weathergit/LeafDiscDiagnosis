# encoding: utf-8
import subprocess
from models.base_model import model_list


def main():
    data_dir = '../datasets/Origin_dataset/'
    for name in model_list.keys():
        print('{} is training'.format(name))
        command = ['python', 'train.py', '--model_name', name,
                   '--data_dir', data_dir]
        subprocess.run(command)
        print('{} is done'.format(name))


if __name__ == "__main__":
    main()
