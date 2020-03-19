#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:42:15 2017

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:19:47 2016

@author: dean
"""
import numpy as np
import os
import sys

#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')
sys.path.append(top_dir +'/nets')

import d_img_process as imp
import d_curve as dc
import scipy.io as  l
import matplotlib.pyplot as plt


s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)
import caffe_net_response as cf

img_n_pix = 64
max_pix_width = [32.,]
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(0, 370)
center_image = round(img_n_pix/2)
x = (center_image-16, center_image+16, 33)
y = (center_image-16, center_image+16, 33)
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                                                                scale=scale,
                                                                x=x,
                                                                y=y)
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict,
                                                        base_stack, 
                                                        npixels=img_n_pix))

#%%
from scipy import misc
save_dir = '/loc6tb/data/images/apc_train/'
f = open(save_dir + 'train.txt', 'w')
point_up = [2,18,22,37,41,57,59,66,69,82,91,70,73,82,90,89,98,100,102,116,114,
            124,126,130,142,148,162,165,167,173,175,178,183,186,189,194,205,
            215,226,228,230,234,236,240,246,252,262,266,274,342,350,358,362,368]  
label = []
for i, img, shape_id in zip(range(len(trans_img_stack)), trans_img_stack, stim_trans_cart_dict['shapes']):
    if i>0:
        f.write('\n')
    fn = save_dir + str(i)  + '.bmp'
    misc.imsave(fn, img)
    f.write(fn)
    if shape_id in point_up:
        f.write(' 1')
        label.append(1)
    else:
        f.write(' -1')
        label.append(-1)
f.close()     
#%%
import os
import numpy as np
import h5py


num_cols = 1
num_rows = len(trans_img_stack)
height = 64
width = 64
total_size = num_cols * num_rows * height * width

data = np.expand_dims(trans_img_stack, 1)
data = data.astype('float32')

label = np.array(label)

with h5py.File('/loc6tb/data/images/' + '/sample_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = label


with open('/loc6tb/data/images' + '/sample_data_list.txt', 'w') as f:
    f.write('/loc6tb/data/images/sample_data.h5\n')

