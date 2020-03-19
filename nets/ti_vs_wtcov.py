#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:21:56 2018

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

def av_cov(da, dims=('unit', 'shapes', 'x', 'y')):
    #get norm cov over last two indices cov ...x n x m --> mxm
    da = da.transpose(*dims)
    try:
        da = da - da.loc(shapes=-1)
        da = da.drop(-1, dim='shapes')
    except:
        print('no baseline, ie no shape indexed as -1')


    resp = da.values
    the_shape = resp.shape
    the_shape = (the_shape[1],) + (np.product(the_shape[2:]),)
    print(the_shape)
    ti_est_all = [norm_cov(unit_resp.reshape(the_shape)) for unit_resp in resp]
    
    return ti_est_all

data_dir = '/loc6tb/'
#%%   
net_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc'
goforit=False       
if 'netwts' not in locals() or goforit:
    da = xr.open_dataset(data_dir + '/data/responses/'+net_name)['resp']
    with open(top_dir + '/nets/netwts.p', 'rb') as f:    
        try:
            netwts = pickle.load(f, encoding='latin1')
        except:
            netwts = pickle.load(f)
#%%
# reshape fc layer to be spatial
    netwts[5][1] = netwts[5][1].reshape((4096, 256, 6, 6))
    wts_by_layer = [layer[1] for layer in netwts]

#%%
    subsamp = 1
    da = da.squeeze()
    da = da.transpose('unit','shapes', 'x', 'y')
    da = da[::subsamp, ...].squeeze() #subsample
    da = da.load()
    da = da - da[:, 0, :, :] #subtract off baseline
    da = da[:, 1:, ...] #get rid of baseline shape   

subsamp = 20    
ti = av_cov(da[::subsamp])
#%%
wtcov = []
for layer in netwts:
    a_layer = layer[1]
    if len(a_layer.shape)>2:
        for unit in a_layer:
            unit = unit.reshape((unit.shape[0],) + (np.product(unit.shape[1:]),) )
            wtcov.append(norm_cov(unit))

plt.scatter(wtcov[::subsamp], ti)

#just need for these to be the same length.  
##%%
##ti_yx, kurt_shapes_yx, kurt_yx, dens, nums, tot_vars_yx = ti_av_cov(da[:, :, :, :])
##%%
#dims = ['unit','chan', 'y', 'x']
#
#netwtsd = {}
#layer_labels = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6']
#for layer, name in zip(wts_by_layer, layer_labels):
#    dim_names = dims[:len(layer.shape)]
#    layer_ind = da.coords['layer_label'].values == name 
#    _ =  da[layer_ind].coords['unit']
#    netwtsd[name] = xr.DataArray(layer, dims=dims, 
#           coords=[range(n) for n in np.shape(layer)])
#    netwtsd[name].coords['unit'] = _
#
#def spatial_opponency(da):
#    da = da.transpose('unit', 'chan', 'y', 'x')
#    data = da.values.reshape(da.shape[:2] + (np.product(da.shape[2:]),))
#    cov = np.matmul(data.transpose(0, 2, 1), data)
#    cov = cov.sum(axis=(1,2)) - np.trace(cov, axis1=1, axis2=2)
#
#    
#    vnorm = np.linalg.norm(data, axis=1)
#    outer_prod = (vnorm[:, :, np.newaxis])*(vnorm[:, np.newaxis, :])
#
#    outer_prod = outer_prod.sum(axis=(1,2)) - np.trace(outer_prod, axis1=1, axis2=2)
#    opponency = cov / outer_prod
#    
#    opponency_da = xr.DataArray(opponency, dims=('unit',))
#    opponency_da.coords['unit'] = da.coords['unit']
#    
#    return opponency_da
#import pandas as pd
#wt_cov_by_layer = []
#for layer_name in zip(layer_labels[1:]):
#    layer = netwtsd[layer_name[0]]
#    print(layer.shape)
#    if len(layer.shape)>2:
#        _ = xr.DataArray(layer, dims=['unit', 'chan', 'x', 'y'])
#        wt_cov = spatial_opponency(_)
#        print(len(wt_cov))
#        wt_cov_by_layer.append(wt_cov)
#wt_covs = np.concatenate(wt_cov_by_layer)
#
#non_k_var = (kurt_shapes_yx<42) * (kurt_shapes_yx>2) * (tot_vars_yx>0) 
#
#keys = ['layer_label', 'unit']
#coord = [da.coords[key].values for key in keys]
#index = pd.MultiIndex.from_arrays(coord, names=keys)
#ti = pd.DataFrame(np.hstack([ti_yx,]), index=index, columns=['ti',])
##%%
#layersbyunit = [[name,]*sum(da.coords['layer_label'].values==name) for name in layer_labels]
#keys = ['layer_label',]
#index = pd.MultiIndex.from_arrays([np.concatenate(layersbyunit),], names=keys)
#wts_covs = pd.DataFrame(np.vstack([wt_covs,]).T, index=index, columns=['wts_cov',])
##%%
#n_plots = len(layer_labels[1:])
#plt.figure(figsize=(12,3))
#
#for i, layer in enumerate(layer_labels[1:]):
#    plt.subplot(1, n_plots, i+1)
#    x = wts_covs.loc[layer]['wts_cov'].values
#    y = np.squeeze(ti.loc[layer].values)
#    if i<4:
#        s=4
#    else:
#        s=1
#    plt.scatter(x, y, s=s, color='k', edgecolors='none')
#    #plt.semilogx()
#    plt.xlim(-0.1,1.02);plt.ylim(-0.1,1.01);
#    if i==0:
#        plt.xlabel('Weight Covariance'); plt.ylabel('T.I.', rotation=0, va='center',ha='right', labelpad=15)
#    if layer == 'conv2':
#        plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['0','','0.5','','1'])
#        plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['0','','0.5','','1'])
#        plt.title(layer + '\nr = ' + str(np.round(np.corrcoef(x,y)[0,1], 2)))
#
#    else:
#        plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['','','','',''])
#        plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['','','','',''])
#        plt.title(layer + '\n' + str(np.round(np.corrcoef(x,y)[0,1], 2)))
#    plt.tight_layout()
#    plt.grid()
