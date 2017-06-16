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
top_dir = top_dir + 'v4cnn'
#%%
net_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc'
net_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(114.0, 114.0, 1)_y_(64, 164, 51)_amp_NonePC370.nc'
data_dir = top_dir
da = xr.open_dataset(data_dir + '/data/responses/'+net_name)['resp']
da = da.squeeze()
#%%


def norm_cov(x, subtract_mean=True):
    #if nxm the get cov nxn
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 0, keepdims=True)
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    vnrm = np.linalg.norm(x, axis=0, keepdims=True)
    denominator = np.sum(np.multiply(vnrm.T, vnrm)[diag_inds]) 
    norm_cov = numerator/denominator
    
    return norm_cov

#import scipy.io as io
#data = 'v4cnn/data/'
#da = io.loadmat(data_dir +  '/data/responses/cadieu_resp.mat')['zzz']
#da = da.reshape(1,65,368)
#da = xr.DataArray(da, dims=['unit', 'x', 'shapes'], 
#                  coords=[[1,],np.arange(0, 65*4, 4), range(368)])
resp = da
stim_width = 32.


    
dims = resp.coords.dims
if ('x' in dims) and ('y' in dims):
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
    stim_in = dist_mat<=(stim_width*1.)
    rf = (resp**2).sum('shapes')>0
    rf[..., :, -1] = False
    rf[..., :, 0] = False
    rf[..., 0, :] = False
    rf[..., -1, :] = False
    rf = rf.values.reshape((rf.shape[0],) + (np.product(rf.shape[1:]),))
    in_spots = stim_in.sum(0)
    overlap = np.array([a_rf * stim_in for a_rf in rf]).sum(-1)
    in_rf = overlap == in_spots[np.newaxis,:]
    
    resp_unrolled = resp.values.reshape(resp.shape[:2] + (np.product(resp.shape[-2:]),))
    ti= []
    for an_in_rf, a_resp in zip(in_rf, resp_unrolled):
        if np.sum(an_in_rf)>2:
            ti.append(norm_cov(a_resp[..., an_in_rf.squeeze()]))
        else:
            ti.append(np.nan)
    
elif ('x' in dims) or ('y' in dims):
    if 'x' in dims:
        resp = resp.transpose('unit', 'shapes', 'x')
        pos = resp.coords['x'].values
        
    elif 'y' in dims:
        resp = resp.transpose('unit', 'shapes', 'y')
        pos = resp.coords['y'].values

    resp = resp - resp[:, 0, ...] #subtract off baseline
    resp = resp[:, 1:, ...] #get rid of baseline shape 
    pos_dist = pos[:, np.newaxis] - pos[:, np.newaxis].T #matrix of differences
    dist_mat = (pos_dist**2)**0.5 #matrix of distances
    stim_in = dist_mat<=(stim_width*1.)#all positions you need to check if responded
    rf = (resp**2).sum('shapes')>0
    #hackish way to make sure test pos is far enough from edge
    #for example if you test two positions close to each other, all adjacent stim
    #are activated but all are on edge, so can't be sure.
    rf[..., 0] = False
    rf[..., -1] = False
    in_rf = rf.copy()
    in_spots = stim_in.sum(0)
    #after overlap only the intersection of stim_in
    #and rf exists so if it is any less then stim_in then not all stim_in points
    #were activated.
    ti = []
    for i, an_rf in enumerate(rf):
        overlap = np.sum(an_rf.values[:, np.newaxis] * stim_in, 0)
        in_pos = overlap == in_spots
        in_rf[i] = in_pos
        
    for an_in_rf, a_resp in zip(in_rf.values, resp.values):
        if np.sum(an_in_rf)>2:
            ti.append(norm_cov(a_resp[..., an_in_rf.squeeze()]))
        else:
            ti.append(np.nan)

resp_av_cov_da = xr.DataArray(ti, coords=resp.coords['unit'].coords)  


#%%
import d_net_analysis as na
ti = na.ti_in_rf(da, stim_width=32)


























































