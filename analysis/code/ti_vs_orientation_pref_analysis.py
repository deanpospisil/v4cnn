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
    ti.append(na.ti_in_rf(da))
#%%  
import pickle
goforit = True      
if 'a' not in locals() or goforit:
    with open(top_dir + 'nets/netwts.p', 'rb') as f:    
        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)
            
wts = a[0][1]
#%%
cnn_name = 'bvlc_reference_caffenet_sinusoids'
da = xr.open_dataset('/loc6tb/' + 'data/responses/' + cnn_name + '.nc')['resp']
cnn_name = 'bvlc_reference_caffenet_cos'
da_cos = xr.open_dataset('/loc6tb/' + 'data/responses/' + cnn_name + '.nc')['resp']

da = da**2 + da_cos**2
#%%
def rfftshift(x):
    y = np.fft.fftshift(x,axes=0)
    return y
layer = 0
unit = 64
ti_x = ti[1]
ti_y = ti[0]
layers = da.coords['layer'].values
layer_labels = da.coords['layer_label'].values
layer_ind = layer == layers
ti_x = np.array(ti_x[:896][layer_ind])
ti_y = np.array(ti_y[:896][layer_ind])
layer_label = layer_labels[layer_ind][0]

x_better = ti_x - ti_y
x_better_ex = np.argsort(x_better)[::-1]
y_better_ex = np.argsort(-x_better)[::-1]
both_good = ti_x + ti_y
both_good_ex = np.argsort(both_good)[::-1]
both_bad_ex = np.argsort(both_good)

cond_name = ['x_better', 'y_better', 'both_good', 'both_bad']
cond_ranks = [x_better_ex, y_better_ex, both_good_ex, both_bad_ex]
n_ranks = 3

for a_cond_name, a_cond_ranks in zip(cond_name, cond_ranks):
    for rank, unit in enumerate(a_cond_ranks[:n_ranks]):
        plt.figure(figsize=(5, 20))
        plt.subplot(511)
        plt.scatter(ti_x, ti_y, s=20, edgecolors='none')
        plt.scatter(ti_x[unit], ti_y[unit], s=40, edgecolors='none')
        plt.xlabel('x');plt.ylabel('y')
        plt.axis('square')
        plt.title('ti_x vs ti_y')
        
        plt.subplot(513)
        l_wt = np.sum(wts[unit]*np.ones(3).reshape(3,1,1), 0)
        plt.imshow(l_wt, cmap=plt.cm.Greys_r)
        plt.xticks([]);plt.yticks([])
        plt.colorbar()
        plt.title('luminance of filter')
        
        plt.subplot(512)
        l_wt = np.sum(wts[unit]*np.ones(3).reshape(3,1,1), 0)
        vis_wts = wts[unit]
        vis_wts = vis_wts-vis_wts.min()
        vis_wts /= vis_wts.max()
        plt.imshow(np.swapaxes(np.swapaxes(vis_wts,0,2),0,1))
        plt.xticks([]);plt.yticks([])
        plt.title('visualized filter')
        #plt.subplot(713)
        #f_resp= da[:, layer_ind]
        #f_resp= f_resp[:, unit].values
        #plt.imshow(rfftshift(f_resp.reshape(227,114)))
        
        plt.subplot(514)
        l_wt_amp = np.abs(np.fft.rfft2(l_wt, s=(227,227)))
        plt.imshow(rfftshift(l_wt_amp))
        plt.title('upsampled fft of filter')
        plt.xlabel('higher freq outward from middle left')
        plt.xlabel('ori rotate around middle left')
        plt.xticks([]);plt.yticks([])
        
#        plt.subplot(615)
#        plt.imshow(np.fft.fftshift(np.rad2deg(np.angle(cart_c[:rft_shape[0], 
#                                                              :rft_shape[1]]))+90., axes=0))
#        plt.colorbar()
        

        #or_powers = []
        #for bin1, bin2 in zip(bins, bins[2:]):
        #    or_inds = (oris>bin1)*(oris<bin2)
        #    or_powers.append(np.sum(f_resp[or_inds]**2))
        #plt.subplot(716)
        #plt.plot(bins[2:], or_powers)
        #plt.title('sinusoids')
        
        
        plt.subplot(515)
        oris = np.rad2deg(np.angle(freq_index)) + 90
        bins = np.linspace(0, 180, 100)
        or_powers = []
        for bin1, bin2 in zip(bins, bins[2:]):
            or_inds = (oris>bin1)*(oris<bin2)
            or_powers.append(np.sum(l_wt_amp.ravel()[or_inds]**2))
        plt.plot(bins[2:], or_powers)
        plt.title('fft')
        plt.xlabel('power at ori')
        plt.tight_layout()
        plt.savefig('/home/dean/Desktop/v4cnn/analysis/figures/images/ori_ti/'
                    + a_cond_name + str(rank))

#%%
mtov=[]
ti_both = []
for unit in range(96):
    l_wt = np.sum(wts[unit]*np.ones(3).reshape(3,1,1), 0)
    mtov.append(np.mean(l_wt)/np.var(l_wt))
    ti_both.append(ti_x[unit] + ti_y[unit])
    
plt.scatter(mtov, ti_both)
plt.xlabel('mean/var')
plt.ylabel('ti_x + ti_y')
    
plt.savefig('/home/dean/Desktop/v4cnn/analysis/figures/images/ori_ti/'
                    + 'DC drives good overall TI')

#%%
da = xr.open_dataset('/loc6tb/' + 'data/responses/' + cnn_name + '.nc')['resp']
   
layer = 4
unit = 3   
ti_x = ti[1]
ti_y = ti[0]

layers = da.coords['layer'].values
layer_labels = da.coords['layer_label'].values
layer_ind = layer == layers
ti_x = np.array(ti_x[layer_ind])
ti_y = np.array(ti_y[layer_ind])
layer_label = layer_labels[layer_ind][0]
plt.subplot(511)
plt.scatter(ti_x, ti_y, s=4, edgecolors='none')
plt.scatter(ti_x[unit], ti_y[unit], s=4, edgecolors='none')
plt.xlabel('x')
plt.ylabel('y')

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