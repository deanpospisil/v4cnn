# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:22:50 2016

@author: dean
"""
import numpy as np
from collections import OrderedDict as ordDict

import os
import sys
import warnings

#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')

sys.path.append('/home/dean/caffe/python')

import d_img_process as dm
from scipy.misc import imread

img_dir = top_dir + '/images/test_imgs_wyeth/'
extension = 'jpg'
all_filenames = os.listdir(img_dir)
img_names = [ file_name for file_name in all_filenames if file_name.split('.')[-1]==extension]
#img_names = dm.load_sorted_dir_numbered_fnms_with_particular_extension( img_dir , 'JPEG')

imgs = [imread(img_dir + img_name).swapaxes(0,-1).swapaxes(1,2)
        for img_name in img_names if len(imread(img_dir + img_name).shape)==3]

rolled = []
rolled = np.array([ np.concatenate([img[2], img[1], img[0]]) for img in imgs]) # need to swap from RGB to BGR

#rolled = rolled - rolled.mean(axis=(0,2,3)).reshape(1, 3, 323, 481)

for i, img in enumerate(rolled):
    np.save(img_dir + str(i) , img)