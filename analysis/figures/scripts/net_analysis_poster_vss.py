# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:16:12 2017

@author: deanpospisil
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
def net_vis_square(da):
    if da.max()>1 or da.max()<0:
        print('Your image is outside the color range [0,1]')
    da = da.transpose('unit', 'y', 'x','chan')
    (m, n) = close_factors(da.shape[0])
    
    data = da.values
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    
    if data.shape[1]<11:
        from scipy.ndimage import zoom
        data_new_size = np.zeros((np.shape(data)[0], 10, 10, 3))
        for i, im in enumerate(data):
            data_new_size[i, ...] = zoom(im, 
                         (2, 2, 1), 
                         order=0)
        data = data_new_size
            
    ypad = xpad = int(data.shape[1]/10)


    padding = ((0, 0), (ypad, ypad), (xpad, xpad), (0, 0))
    data = np.pad(data, padding, mode='constant', constant_values=0.9)
    #data = data.reshape(m*data.shape[1], n*data.shape[2], data.shape[3], order='C')
        # tile the filters into an image
    data = data.reshape((m, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    data = data.reshape((m * data.shape[1], n * data.shape[3], data.shape[4]))
    
    ax = plt.subplot(111)
    ax.imshow(data, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[pos].set_visible(False) for pos in ['left','right','bottom','top']]

    return ax, data
#%%
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

net_resp_name = 'bvlc_reference_caffenet_nat_image_resp_371.nc'
da = xr.open_dataset(top_dir + '/data/responses/' + net_resp_name)['resp']
#%%
from more_itertools import unique_everseen
layer_num = da.coords['layer']
layer_label_ind = da.coords['layer_label'].values
split_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',]
dims = ['unit','chan', 'y', 'x']
layer_names = list(unique_everseen(layer_label_ind))
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6','fc7','fc8']

netwtsd = {}
for layer, name in zip(wts_by_layer,layer_names):
    dim_names = dims[:len(layer.shape)]
    netwtsd[name] = xr.DataArray(layer, dims=dim_names)


conv1 = netwtsd['conv1']
conv1vis = conv1 - conv1.min(['chan', 'y', 'x'])
conv1vis = conv1vis/conv1vis.max(['chan', 'y', 'x'])
#conv1vis = conv1vis/conv1vis.max()

#conv1vis = conv1vis[:, :, :5, :5]
ax, data = net_vis_square(conv1vis)
plt.savefig(top_dir + '/analysis/figures/images/early_layer/1st_layer_filters.pdf')

#%%
def variance_to_power_ratio(da):
    red_dims = list(set(da.dims) - set('unit'))
    var = ((da-da.mean('chan'))**2).sum(red_dims)
    pwr =  (da**2).sum(red_dims)
    return var/pwr

def receptive_field(da):
    return (da**2).sum(['x','y'])

def princ_comp_pwr_maps()
    
    


    
    