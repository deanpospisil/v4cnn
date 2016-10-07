# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:45:56 2016

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:35:06 2016

@author: dean
"""
import os
import sys
import warnings
import numpy as np
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')
import xarray as xr
import re

pre='bvlc_caffenet_reference_shuffle_layer_'
post = 'APC362_pix_width[30.0]_pos_(64.0, 164.0, 51).nc'

resp_list = [pre + str(layer) +post for layer in range(8)]
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
layer = 0
all_resp_list= []
while layer<=2:
    cnn = xr.open_dataset(top_dir + 'data/responses/' + resp_list[layer])
    cnn = cnn.squeeze().transpose('unit', 'shapes','x')
    same_layer = [re.findall('/d', unit_name) == [str(layer+1)] for unit_name in cnn.coords['unit'].layer_label.values]
    all_resp_list.append(cnn.isel(unit=same_layer))
    layer += 1
