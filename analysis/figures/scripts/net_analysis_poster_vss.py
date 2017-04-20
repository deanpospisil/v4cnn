# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:16:12 2017

@author: deanpospisil
"""

import os, sys
import numpy as np
import matplotlib as plt
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir + 'v4cnn'
import xarray as xr

def close_factors(n):
    factor_list = []
    for n_in in range(1,n):
        if (n%n_in) == 0:
            factor_list.append(n_in)
    factor_array = np.array(factor_list)
    paired_factors = np.array([factor_array, n/factor_array])
    paired_factors.shape
    best_ind = np.argmin(abs(paired_factors[1]-paired_factors[0]))
    closest_factors = paired_factors[:,best_ind]
    return closest_factors[0], closest_factors[1]


import pickle
goforit=False       
if 'netwts' not in locals() or goforit:
    with open(top_dir + '/nets/netwts.p', 'rb') as f:    
        try:
            netwts = pickle.load(f, encoding='latin1')
        except:
            netwts = pickle.load(f)
# reshape fc layer to be spatial
netwts[5][1] = netwts[5][1].reshape((4096, 256, 6, 6))
wts_by_layer = [layer[1] for layer in netwts]
#%%
net_resp_name = 'bvlc_reference_caffenet_nat_image_resp_371.nc'
da = xr.open_dataset(top_dir + '/data/responses/' + net_resp_name)['resp']
layer_num = da.coords['layer']
layer_label = da.coords['layer_label']


