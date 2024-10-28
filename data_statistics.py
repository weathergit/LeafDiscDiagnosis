# encoding:utf-8
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from glob import glob
from shapely import Point, Polygon
from shapely.predicates import intersects,contains


def counting_instance(path):
    with open(path, 'rb') as f:
        label_file = json.load(f)
    label_df = pd.DataFrame.from_dict(label_file['shapes'])
    labels = label_df['label'].unique()
    # print(labels)
    # print(len(labels))
    # print(label_df)
    # label_df.to_csv('test.csv',encoding='utf-8')

    results = []


    if len(labels) == 1:
        healthy_instance = label_df.shape[0]
        results = np.zeros(healthy_instance)
    else:
        disc_df = label_df[label_df['label']=='disc']
        downy_df = label_df[label_df['label']=='downy']

        # loop for the disc
        for dl, dt in zip(disc_df['points'],disc_df['shape_type']):
            # some bugs may occurs here. 
            if dt == 'circle':
                x1,y1,x2,y2 = dl[0][0],dl[0][1],dl[1][0],dl[1][1]
                xy_sum = (x1-x2)**2 + (y1-y2)**2
                radius = np.sqrt(xy_sum)
                disc_shp =  Point(x1,y1).buffer(radius)
                
            elif dt == 'polygon':
                disc_shp = Polygon(dl)
            else:
                print('Unknown shapr of disc')
            
            disc_area = disc_shp.area
            # print(disc_area)
            # loop for the intersects of disc and downy
            downy_area = 0
            for yl, yt in zip(downy_df['points'],downy_df['shape_type']):
                if yt == 'polygon':
                    downy_shp = Polygon(yl)
                    downy_center = downy_shp.centroid
                    # if intersects(disc_shp, downy_shp):
                    if disc_shp.contains(downy_center):
                        # print(downy_area)                        
                        downy_area += downy_shp.area 
                    else:
                        pass
                else:
                    print('Unknown shape of downy')
            diseases_rate = downy_area / disc_area
            results.append(diseases_rate)
    return results
            

def main():
    json_files = sorted(glob('./datasets/images/*json'))
       
    fns_list = []
    results = []
    zy_downy = []
    for fn in json_files:
        print(fn)
        try:
            res = counting_instance(fn)
            if len(res) == 1:
                zy_downy.extend(res)
            results.extend(res)
            fns_list.extend([fn]*len(res))
        except:
            print(fn,'    wrong')
    assert len(fns_list) == len(results)
    df = pd.DataFrame.from_dict({'files': fns_list,'severity':results})
    df1 = pd.Series(zy_downy)
    
    df.to_csv('./counting.csv',encoding='utf-8')
    df1.to_csv('./counting_zy.csv',encoding='utf-8')


if __name__ == '__main__':
    main()
    # a=counting_instance('./datasets/images/IMG_0002zy (116).json')
    # print(a)