#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:59:41 2017

@author: dean
"""
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
#%%
def ti_av_cov(da):
    dims = da.coords.dims
    #get the da in the right shape
    if ('x' in dims) and ('y' in dims):
        da = da.transpose('unit','shapes', 'x', 'y')
    elif ('x' in dims):
        da = da.transpose('unit', 'shapes', 'x')
    elif ('y' in dims):
        da = da.transpose('unit', 'shapes', 'y')
        
    #some data to store
    ti = np.zeros(np.shape(da)[0])
    dens = np.zeros(np.shape(da)[0])
    nums = np.zeros(np.shape(da)[0])
    tot_vars = np.zeros(np.shape(da)[0])
    kurt_shapes = np.zeros(np.shape(da)[0])
    kurt_x =  np.zeros(np.shape(da)[0])

    for i, unit_resp in enumerate(da):
        if len(unit_resp.shape)>2:
            #unwrap spatial
            unit_resp = unit_resp.values.reshape(unit_resp.shape[0], unit_resp.shape[1]*unit_resp.shape[2])   
        else:
            unit_resp = unit_resp.values
        unit_resp = unit_resp.astype(np.float64)
        unit_resp = unit_resp - np.mean(unit_resp, 0, keepdims=True, dtype=np.float64)
 

        cov = np.dot(unit_resp.T, unit_resp)
        cov[np.diag_indices_from(cov)] = 0
        numerator = np.sum(np.triu(cov))

        vlength = np.linalg.norm(unit_resp, axis=0, keepdims=True)
        max_cov = np.outer(vlength.T, vlength)
        max_cov[np.diag_indices_from(max_cov)] = 0
        denominator= np.sum(np.triu(max_cov))

        kurt_shapes[i] = kurtosis(np.sum(unit_resp**2, 1))
        kurt_x[i] = kurtosis(np.sum(unit_resp**2, 0))
        den = np.sum(max_cov)
        num = np.sum(cov)
        dens[i] = den
        nums[i] = num
        tot_vars[i] = np.sum(unit_resp**2)
        if den!=0 and num!=0:
            ti[i] = num/den 
    return ti, kurt_shapes, kurt_x, dens, nums, tot_vars 

data_dir = '/loc6tb/'
#%%   
goforit=False       
if 'netwts' not in locals() or goforit:
    with open(top_dir + '/nets/netwts.p', 'rb') as f:    
        try:
            netwts = pickle.load(f, encoding='latin1')
        except:
            netwts = pickle.load(f)
#%%
# reshape fc layer to be spatial
netwts[5][1] = netwts[5][1].reshape((4096, 256, 6, 6))
wts_by_layer = [layer[1] for layer in netwts]
subsamp = 10 

net_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc'
da = xr.open_dataset(data_dir + '/data/responses/'+net_name)['resp']

#%%
subsamp = 10
da = da.squeeze()
da = da.transpose('unit','shapes', 'x', 'y')
da = da[::subsamp, ...].squeeze() #subsample
da = da.load()
da = da - da[:, 0, :, :] #subtract off baseline
da = da[:, 1:, ...] #get rid of baseline shape   
from scipy.stats import kurtosis

#%%
ti_yx, kurt_shapes_yx, kurt_yx, dens, nums, tot_vars_yx = ti_av_cov(da[:, :, :, :])
#%%
dims = ['unit','chan', 'y', 'x']

netwtsd = {}
layer_labels = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6']
for layer, name in zip(wts_by_layer, layer_labels):
    dim_names = dims[:len(layer.shape)]
    layer_ind = da.coords['layer_label'].values == name 
    _ =  da[..., layer_ind].coords['unit']
    netwtsd[name] = xr.DataArray(layer, dims=dims, 
           coords=[range(n) for n in np.shape(layer)])
    netwtsd[name].coords['unit'] = _

def spatial_opponency(da):
    da = da.transpose('unit', 'chan', 'y', 'x')
    data = da.values.reshape(da.shape[:2] + (np.product(da.shape[2:]),))
    cov = np.matmul(data.transpose(0, 2, 1), data)
    cov = cov.sum(axis=(1,2)) - np.trace(cov, axis1=1, axis2=2)

    
    vnorm = np.linalg.norm(data, axis=1)
    outer_prod = (vnorm[:, :, np.newaxis])*(vnorm[:, np.newaxis, :])

    outer_prod = outer_prod.sum(axis=(1,2)) - np.trace(outer_prod, axis1=1, axis2=2)
    opponency = cov / outer_prod
    
    opponency_da = xr.DataArray(opponency, dims=('unit',))
    opponency_da.coords['unit'] = da.coords['unit']
    
    return opponency_da
import pandas as pd
wt_cov_by_layer = []
for layer, layer_name in zip(netwtsd, layer_labels):
    print(layer[1].shape)
    if len(layer[1].shape)>2:
        _ = xr.DataArray(layer[1], dims=['unit', 'chan', 'x', 'y'])
        wt_cov = spatial_opponency(_)
        print(len(wt_cov))
        wt_cov_by_layer.append(wt_cov)
wt_covs = np.concatenate(wt_cov_by_layer)

non_k_var = (kurt_shapes_yx<42) * (kurt_shapes_yx>2) * (tot_vars_yx>0) 

keys = ['layer_label', 'unit']
coord = [da.coords[key].values for key in keys]
index = pd.MultiIndex.from_arrays(coord, names=keys)
ti = pd.DataFrame(np.hstack([ti_yx,]), index=index, columns=['ti',])
#%%
layersbyunit = [[name,]*sum(da.coords['layer_label'].values==name) for name in layer_labels]
keys = ['layer_label',]
index = pd.MultiIndex.from_arrays([np.concatenate(layersbyunit),], names=keys)
wts_covs = pd.DataFrame(np.vstack([wt_covs,]).T, index=index, columns=['wts_cov',])
#%%
n_plots = len(layer_labels[1:])
plt.figure(figsize=(12,3))

for i, layer in enumerate(layer_labels[1:]):
    plt.subplot(1, n_plots, i+1)
    x = wts_covs.loc[layer]['wts_cov'].values
    y = np.squeeze(ti.loc[layer].values)
    if i<4:
        s=4
    else:
        s=1
    plt.scatter(x, y, s=s, color='k', edgecolors='none')
    #plt.semilogx()
    plt.xlim(-0.1,1.02);plt.ylim(-0.1,1.01);
    if i==0:
        plt.xlabel('Weight Covariance'); plt.ylabel('T.I.', rotation=0, va='center',ha='right', labelpad=15)
    if layer == 'conv2':
        plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['0','','0.5','','1'])
        plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['0','','0.5','','1'])
        plt.title(layer + '\nr = ' + str(np.round(np.corrcoef(x,y)[0,1], 2)))

    else:
        plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['','','','',''])
        plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['','','','',''])
        plt.title(layer + '\n' + str(np.round(np.corrcoef(x,y)[0,1], 2)))
    plt.tight_layout()
    plt.grid()
