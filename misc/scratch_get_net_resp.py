#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:41:57 2018

@author: dean
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:46:11 2017

@author: dean
"""
import numpy as np
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir + 'v4cnn'
import xarray as xr
import pandas as pd


import pickle
goforit=True      
if 'netwts' not in locals() or goforit:
    with open(top_dir + '/nets/netwts.p', 'rb') as f:    
        try:
            netwts = pickle.load(f, encoding='latin1')
        except:
            netwts = pickle.load(f)
# reshape fc layer to be spatial
netwts[5][1] = netwts[5][1].reshape((4096, 256, 6, 6))
wts_by_layer = [layer[1] for layer in netwts]

data_dir = '/loc6tb'
net_resp_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370.nc'
da = xr.open_dataset(data_dir + '/data/responses/' + net_resp_name)['resp']
if not type(da.coords['layer_label'].values[0]) == str:
    da.coords['layer_label'].values = [thing.decode('UTF-8') 
    for thing in da.coords['layer_label'].values]
if not type(da.coords['layer_label'].values[0]) == str:
    da.coords['layer_label'].values = [str(thing) 
    for thing in da.coords['layer_label'].values]
da.coords['unit'] = range(da.shape[-1])

from more_itertools import unique_everseen
layer_num = da.coords['layer']
layer_label_ind = da.coords['layer_label'].values
split_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',]
dims = ['unit','chan', 'y', 'x']
layer_names = list(unique_everseen(layer_label_ind))
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6',]


netwtsd = {}
for layer, name in zip(wts_by_layer, layer_names):
    dim_names = dims[:len(layer.shape)]
    layer_ind = da.coords['layer_label'].values == name 
    unit_coords =  da[..., layer_ind].coords['unit']
    netwtsd[name] = xr.DataArray(layer, dims=dims, 
           coords=[range(n) for n in np.shape(layer)])
    netwtsd[name].coords['unit'] = unit_coords.values
    
    


#%%
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ann_fn = 'blvc_caffenet_iter_1'
sys.path.append('/home/dean/caffe/python')

import caffe
net_proto_name = ann_dir + 'deploy_fixing_relu_saved.prototxt'
net_wts_name = ann_dir + ann_fn + '.caffemodel'
net = caffe.Net(net_proto_name, net_wts_name, caffe.TEST)

net.params['conv1'][0].data[...] = netwtsd['conv1'][..., :8, :8]


net.save(ann_dir + 'v4_model.caffemodel')
        

