import os
import nibabel as nib
from matplotlib import pyplot
from PIL import Image
import numpy as np
import cv2

path = (
"/home/sergey/PycharmProjects/Pytorch-UNet/MICCAI_BraTS17_Data_Training/HGG", "/home/sergey/PycharmProjects/Pytorch-UNet/MICCAI_BraTS17_Data_Training/LGG")

for i in path:
    for j in os.listdir(i):
        for k in os.listdir(i + '/' + j):
            if k.find('flair') != -1:

                img = cv2.normalize(nib.load(i + '/' + j + '/' + k))

                for i in img[:, :, 80]:
                    a = Image.fromarray(img[:,:,80])

            if k.find('flair') != -1:
                img = nib.load(i + '/' + j + '/' + k).get_data()
