#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:56:19 2017

@author: dean
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+'xarray/')
#%%
import xarray as xr
top_dir = top_dir + '/v4cnn'
#%%
#net_name = 'bvlc_reference_caffenetpix_width[25.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc'
net_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370.nc'

data_dir = top_dir


da = xr.open_dataset(data_dir + '/data/responses/'+net_name)['resp']
#%%

da = da[:100]
#%%
da = da.squeeze()

#%%
rf = (da**2).sum('shapes')>0
#%%
resp = da
stim_diam = 32.
dims = resp.coords.dims

if ('x' in resp) and ('y' in dims):
    resp = resp.transpose('unit','shapes', 'x', 'y')
    resp = resp - resp[:, 0, :, :] #subtract off baseline
    resp = resp[:, 1:, ...] #get rid of baseline shape 
    
    x = resp.coords['x'].values
    y = resp.coords['y'].values
    
    x_grid = np.tile(x, (len(y), 1)).ravel()
    y_grid = np.tile(y[:, np.newaxis], (1, len(x))).ravel()
    
    x_dist = x_grid[:, np.newaxis] - x_grid[:, np.newaxis].T
    y_dist = y_grid[:, np.newaxis] - y_grid[:, np.newaxis].T
    
    dist_mat = (x_dist**2 + y_dist**2)**0.5

    stim_in = dist_mat<=(stim_diam*1.5)
        
elif ('x' in dims):
    resp = resp.transpose('unit', 'shapes', 'x')

    resp = resp - resp[:, 0, ...] #subtract off baseline
    resp = resp[:, 1:, ...] #get rid of baseline shape 
    
    x = resp.coords['x'].values
    x_dist = x[:, np.newaxis] - x[:, np.newaxis].T #matrix of differences
    dist_mat = (x_dist**2)**0.5 #matrix of distances
    stim_in = dist_mat<=(stim_diam*1.)#all positions you need to check if responded
    
    rf = (resp**2).sum('shapes')>0 #get where there was a response.
        #hackish way to make sure test pos is far enough from edge
    #for example if you test two positions close to each other, all adjacent stim
    #are activated but all are on edge, so can't be sure.
    rf[..., 0] = False
    rf[..., -1] = False
    in_rf = rf.copy()
    for i, an_rf in enumerate(rf):
        in_and_on = np.sum(an_rf.values[:, np.newaxis] * stim_in, 0)
        in_spots = stim_in.sum(0)
        in_pos = in_and_on == in_spots
        in_rf[i] = in_pos

elif ('y' in dims):
    resp = resp.transpose('unit', 'shapes', 'y')
    resp_vals = resp.values
plt.plot(in_pos)
#%%
x_grid = np.tile(x, (len(y), 1)).ravel()
y_grid = np.tile(y[:, np.newaxis], (1, len(x))).ravel()
x_dist = x_grid[:, np.newaxis] - x_grid[:, np.newaxis].T
y_dist = y_grid[:, np.newaxis] - y_grid[:, np.newaxis].T

dist_mat = (x_dist**2 + y_dist**2)**0.5
stim_diam = 32
stim_in = dist_mat<(stim_diam*1.5)
#%%
in_rf_num = []
for an_rf in rf[:10000]:
    resp_plus_close = an_rf.values.ravel()[:, np.newaxis] * stim_in
    in_rf = np.sum(resp_plus_close, 0) == np.sum(stim_in, 0)
    in_rf_num.append(sum(in_rf))
plt.plot(in_rf_num)
    