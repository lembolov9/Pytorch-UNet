import json
import random

import numpy as np
from os import listdir
from PIL import Image
from matplotlib import pyplot
p = '/home/sergey/PycharmProjects/Image_segmentation/'
paths = ("TRAIN_SET/Fedor_no_annot/Sag-CUBE-T2", "TRAIN_SET/Jane/AX-T2-FLAIR-01-03-17", "TRAIN_SET/Jane/AX-T2-FLAIR-24-09-18",
         "TRAIN_SET/Marianne/10-09-17/AX-FSE-T2-10-09-17", "TRAIN_SET/Marianne/10-09-17/AX-T2-FLAIR-10-09-17", "TRAIN_SET/Marianne/25-12-16/AX-FSE-T2-25-12-16",
         "TRAIN_SET/Marianne/25-12-16/AX-T1-SE+C-25-12-16", "TRAIN_SET/Marianne/25-12-16/AX-T2-FLAIR-25-12-16")
filenames_train = []
filenames_target = []
for i in paths:
    filenames_train.append(sorted(listdir(i+'/scans/'), key= lambda x: int(x.split('.')[-2])))
    filenames_target.append(sorted(listdir(i+'/rois/'), key= lambda x: int(x.split('.')[-2])))

X_train = []
X_target = []

f = open("std_mean_dict.json", "r")

p = "/home/sergey/PycharmProjects/Pytorch-UNet/result"

std_mean_dict = json.load(f)
print(std_mean_dict)

for k, i in enumerate(filenames_train):
    test_array = []
    for s, j in enumerate(i):

        print(i)
        im = Image.open(paths[k] + '/scans/' + j)
        im2 = Image.open(paths[k] + '/rois/' + j)
        print(np.asarray(im).dtype)
        im = im.resize((240, 240), Image.LANCZOS)
        im2 = im2.resize((240, 240), Image.LANCZOS)
        im2 = im2.convert('L')
        if random.random() > 0.8:
            im.save(p + '/test/' + paths[k].replace('/', '_') + j)
            im2.save(p + '/test_y/' + paths[k].replace('/', '_') + j)
        else:
            im.save(p+'/train/'+paths[k].replace('/','_')+j)
            im2.save(p + '/target/' + paths[k].replace('/', '_') + j)
        print(np.asarray(im).shape)
