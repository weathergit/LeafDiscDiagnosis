import json
from shapely import Polygon
import pandas as pd
import os
import glob

def json2coco(json_path):
    """convert  annoated json file into xyxy/ xywh format to match YOLO input
    Args:
        json_path (str): json path.
    """
    with open(json_path,'rb') as f:
        fn = json.load(f)
    w, h = fn.get('imageWidth'), fn.get('imageHeight')
    df = pd.DataFrame(fn['shapes'])
    for label, points in zip(df['label'], df['points']):
        if label == 'disc':
            shape = Polygon(points)
            extent = shape.bounds
            x1,y1, x2, y2 = extent
            x_center, y_center = (x1+x2)/(2*w), (y1+y2)/(2*h)
            box_w, box_h = (x2-x1) / w, (y2-y1) / h
            text_box = '0 {} {} {} {}'.format(x1, y1, x2, y2)
            text_yolo = '0 {} {} {} {}'.format(x_center, y_center, box_w, box_h)
            dir_name = os.path.dirname(json_path)
            basename = os.path.basename(json_path)
            outname_box = './new_datasets/new_label_xyxy/' + basename[:-4] + 'txt'
            outname_yolo = './new_datasets/new_label_xywh/' + basename[:-4] + 'txt'
            with open(outname_box, 'a+') as f:
                f.write(text_box)
                f.write('\n')
            with open(outname_yolo, 'a+') as f:
                f.write(text_yolo)
                f.write('\n')


def main():
    fns = glob.glob('./new_datasets/leaf_disc/*json')
    # fns = ['./new_datasets/leaf_disc/IMG_9388.json']
    # print(len(fns))
    for fn in fns:
        print(fn)
        json2coco(fn)
        # try:
        #     json2coco(fn)
        # except:
        #     print(fn)


def check_json():
    fns = glob.glob('./new_datasets/leaf_disc/*json')
    fns = [os.path.basename(b)[:-5] for b in fns]

    txts = [b[:-4] for b in os.listdir('./new_datasets/new_label_xyxy/')]
    for a, b in zip(fns, txts):
        print(a, b)
        assert a == b


if __name__ == '__main__':
    main()