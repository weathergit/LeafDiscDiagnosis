# encoding: utf-8

import subprocess


def main():
    """
    use 'labelme2voc.py' from YOLO repository. all codes for instance segmenatation was uncommented.
    create mask png and mask array for original dataset, based on Json file。
    """
    script = './2-labelme2voc.py'
    inputdir = '../new_datasets/2.leaf_disc/'
    outputdir = '../new_datasets/3.3new_datasets_mask/'
    labels = '../labels.txt'
    subprocess.run(['python',script,inputdir, outputdir,'--labels',labels],shell=True)

if __name__ == '__main__':
    main()