# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:55:15 2017

@author: deanpospisil
"""

import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
top_dir = os.getcwd().split('v4cnn')[0]
top_dir = top_dir + 'v4cnn'


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

net_resp_name = 'bvlc_reference_caffenety_test_APC362_pix_width[32.0]_x_(104.0, 124.0, 11)_x_(104.0, 124.0, 11)_amp_None.nc'
da = xr.open_dataset(top_dir + '/data/responses/' + net_resp_name)['resp']
if not type(da.coords['layer_label'].values[0]) == str:
    da.coords['layer_label'].values = [thing.decode('UTF-8') for thing in da.coords['layer_label'].values]
da.coords['unit'] = range(da.shape[-1])

dims = ['unit','chan', 'y', 'x']
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6',]

from collections import OrderedDict
netwtsd = OrderedDict()
for layer, name in zip(wts_by_layer, layer_names):
    dim_names = dims[:len(layer.shape)]
    layer_ind = da.coords['layer_label'].values == name 
    _ =  da[..., layer_ind].coords['unit']
    netwtsd[name] = xr.DataArray(layer, dims=dims, 
           coords=[range(n) for n in np.shape(layer)])
    netwtsd[name].coords['unit'] = _ 
pickle.dump(netwtsd, open(top_dir + '/nets/netwtsd.p', "wb" ) )

goforit=True      
if 'netwtsd' not in locals() or goforit:
    with open(top_dir + '/nets/netwtsd.p', 'rb') as f:    
        try:
            netwtsd = pickle.load(f, encoding='latin1')
        except:
            netwtsd = pickle.load(f)
    
    