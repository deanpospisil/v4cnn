#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:01:30 2017

@author: dean
"""

import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn')
sys.path.insert(0, top_dir + 'xarray/');
top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr
import pandas as pd
cnn_name = 'bvlc_reference_caffenet_sinusoids'
data_dir = '/loc6tb/'


imshape = (227, 227)
rft = np.fft.rfft2(np.ones(imshape))
rft_shape = rft.shape

r = np.fft.fftfreq(imshape[0]).reshape((imshape[0]), 1)
c = np.fft.fftfreq(imshape[1]).reshape((1, imshape[1]))

cart_c = c + r*1j
freq_index = cart_c[:rft_shape[0], :rft_shape[1]].ravel()

#%%
import d_net_analysis as na
y_nm = 'bvlc_reference_caffenetpix_width[32.0]_x_(114.0, 114.0, 1)_y_(64, 164, 51)_amp_NonePC370.nc'
x_nm = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370.nc'
ti = []
k = []
for net_name in [y_nm, x_nm]:
    da = xr.open_dataset(data_dir + '/data/responses/'+ net_name)['resp'].squeeze()
    k.append(na.kurtosis_da(da))
    ti.append(na.ti_in_rf(da, stim_width=32))
#%%  
da = xr.open_dataset('/loc6tb/' + 'data/responses/' + cnn_name + '.nc')['resp']
      
ti_x = ti[1]
ti_y = ti[0]

layers = da.coords['layer'].values
layer_labels = da.coords['layer_label'].values

layer = 4
layer_ind = layer == layers
layer_label = layer_labels[layer_ind][0]
plt.scatter(np.array(ti_x[layer_ind]), np.array(ti_y[layer_ind]),s=4, edgecolors='none')
conv2_ti_x = ti_x[layer_ind]
conv2_ti_y = ti_y[layer_ind]

f_resp= da[:, layer_ind]
f_resp= f_resp[:, 100].values

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.imshow(f_resp.reshape(227,114))
plt.colorbar()
plt.subplot(212)
plt.imshow(np.abs(cart_c[:rft_shape[0], :rft_shape[1]]))

bins = np.linspace(0, 180, 100)
or_powers = []
oris = np.rad2deg(np.angle(freq_index)) + 90

for bin1, bin2 in zip(bins, bins[2:]):
    or_inds = (oris>bin1)*(oris<bin2)
    or_powers.append(np.sum(f_resp[or_inds]**2))
plt.figure() 
plt.plot(bins[2:],or_powers)

plt.figure()
plt.imshow(np.rad2deg(np.angle(cart_c[:rft_shape[0], :rft_shape[1]]))+90)
plt.colorbar()