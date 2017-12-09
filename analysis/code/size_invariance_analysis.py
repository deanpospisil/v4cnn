#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:30:34 2017

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
import xarray as xr
da = xr.open_dataset('/loc6tb/data/responses/bvlc_reference_caffenetpix_width(8, 16, 10)_x_(114.0, 114.0, 1)_y_(114.0, 114.0, 1)PC370.nc')['resp']


def norm_cov(x, subtract_mean=True):
    
    #if nxm the get cov mxm
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 0, keepdims=True)
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    vnrm = np.linalg.norm(x, axis=0, keepdims=True)
    denominator = np.sum(np.multiply(vnrm.T, vnrm)[diag_inds]) 
    norm_cov = numerator/denominator
    
    return norm_cov

def av_cov(da, dims=('unit', 'shapes', 'scale')):
    #get norm cov over last two indices cov ...x n x m --> mxm
    da = da.transpose(*dims)
    try:
        da = da - da.loc(shapes=-1)
        da = da.drop(-1, dim='shapes')
    except:
        print('no baseline, ie no shape indexed as -1')


    resp = da.values

    ti_est_all = [norm_cov(unit_resp) for unit_resp in resp]
    
    return ti_est_all

scale_inv = av_cov(da.squeeze())
da_an = xr.DataArray(scale_inv, coords=[da.coords['unit'],])
da_an['layer'] = da.coords['layer']

#%%

layer_label = list(da.coords['layer_label'].values)
layers = sorted(set(layer_label), key=lambda x: layer_label.index(x))
meds = da_an.groupby('layer', squeeze=False).median()
var = da_an.groupby('layer', squeeze=False).std()

plt.errorbar(range(21), meds, yerr=var)
plt.xticks(range(21), rotation='vertical')
plt.gca().set_xticklabels(layers)


def norm_avcov_iter(x, subtract_mean=True):
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 1, keepdims=True)
    diag_inds = np.triu_indices(x.shape[-1], k=1)
    numerator = [np.sum(np.dot(unit.T, unit)[diag_inds]) for unit in x]
    
    vnrm = np.linalg.norm(x, axis=1, keepdims=True)
    denominator = [np.sum(np.multiply(unit.T, unit)[diag_inds]) for unit in vnrm]    
    norm_cov = np.array(numerator)/np.array(denominator)
    norm_cov[np.isnan(norm_cov)] = 0
    
    return norm_cov