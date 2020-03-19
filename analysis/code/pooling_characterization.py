#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:22:37 2017

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
    
def unit_norm(a, dims):
    a = a - a.mean(dims)
    a = a / (a**2).sum(dims)**0.5
    return a

def cor_over_dim(a, b, dims):
    a = unit_norm(a, dims)
    b = unit_norm(b, dims)
    r = (a*b).sum(dims)
    return r


data_dir = '/loc6tb/'
fns  = ['v4cnn/bvlc_reference_caffenetpix_width[32.0]_x_(113.0, 113.0, 1)_y_(113.0, 113.0, 1)_offsets_fine_PC370'
 ,'v4cnn/blvc_caffenet_iter_1pix_width[32.0]_x_(113.0, 113.0, 1)_y_(113.0, 113.0, 1)_offsets_fine_PC370']
#%%
#need to reduce size of dataarray according to when there are no responses.
labels = ['trained', 'untrained']
for fn, label in zip(fns, labels):
    data_dir = '/loc6tb/'
    da = xr.open_dataset(data_dir + 'data/responses/' + fn + '.nc')['resp'].squeeze()
    shapes2 = da.sel(shapes=-1)
    shapes1 = da.sel(shapes2=-1)
    shapes2, shapes1 =xr.broadcast(shapes2, shapes1)
    
    b = xr.concat([shapes1,shapes2], dim='lr')
    import d_net_analysis as na
    
    
    max_hat = b.max('lr')
    sum_hat = b.sum('lr')
    
    lin_hat = da.copy()
    for unit in range(22096):
        for offset in range(10):
            a = b.isel(unit=unit, offsetsx=offset).squeeze().transpose(
                                                'lr','shapes', 'shapes2')
            al = a.isel(lr=0).sum('shapes2')
            ar = a.isel(lr=1).sum('shapes')
            
            
            a = a.values.reshape(2, 31**2).T
            B = da.isel(unit=unit, offsetsx=offset).squeeze().transpose('shapes', 'shapes2').values
            B = B.reshape(31**2, 1)
            non_zero_ind = a==0
            a = a - np.mean(a, 0, keepdims=True)
            B = B - np.mean(B, 0, keepdims=True)

            x = np.linalg.lstsq(a,B)[0]
            l_hat = np.dot(a, x)
            l_hat = l_hat.reshape(31, 31)
            lin_hat[dict(unit=unit, offsetsx=offset)] = l_hat
            
    
    
    plt.figure()
    da.isel(unit=6000, offsetsx=2).plot()
    plt.figure()
    max_hat.isel(unit=6000, offsetsx=2).plot()
    dims = ['shapes', 'shapes2']

    lpool = cor_over_dim(lin_hat.copy(), da, dims)
    mpool = cor_over_dim(max_hat.copy(), da, dims) 
    spool = cor_over_dim(sum_hat.copy(), da, dims) 
    
    data_dir = '/loc6tb/data/an_results/'
    spool.to_dataset().to_netcdf(data_dir + '/'+ label + '_sumpool_fit_trained_32pix_3offset.nc')
    mpool.to_dataset().to_netcdf(data_dir + '/'+ label + '_maxpool_fit_trained_32pix_3offset.nc')
    lpool.to_dataset().to_netcdf(data_dir + '/'+ label + '_linpool_fit_trained_32pix_3offset.nc')
#%%
from scipy.stats import kurtosis
k=[]
for frames in da.values.T:
    k.append([kurtosis(frame.ravel(), nan_policy='omit') for frame in frames])


#%%
#da = xr.open_dataset(data_dir + 'data/responses/' + fn + '.nc')['resp'].squeeze()
label = 'trained'
data_dir = '/loc6tb/data/an_results'
spool = xr.open_dataset(data_dir + '/'+ label + '_sumpool_fit_trained_32pix_3offset.nc')['resp']
mpool = xr.open_dataset(data_dir + '/'+ label + '_maxpool_fit_trained_32pix_3offset.nc')['resp']
lpool = xr.open_dataset(data_dir + '/'+ label + '_linpool_fit_trained_32pix_3offset.nc')['resp']

k = np.array(k)
lpool = lpool[:, np.sum(k,1)<20]
mpool = mpool[:, np.sum(k,1)<20]
spool = spool[:, np.sum(k,1)<20]
#%%
layers_to_examine = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
layers_to_examine = ['conv1', 'relu1', 'pool1', 'norm1', 'conv2', 'relu2', 'pool2',
                     'norm2', 'conv3', 'relu3', 'conv4', 'relu4', 'conv5', 'relu5',
                     'pool5', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8']

#layers_to_examine = np.unique(mpool.coords['layer_label'].values)
for layer in layers_to_examine:
    layer_ind = (mpool.coords['layer_label'] == layer).values
    #layer_ind = layer_ind & da.std(['shapes', 'shapes2', 'offsetsx'])
    colors = ['r', 'g', 'b']
    plt.figure()
    for i, color in enumerate(colors):
        plt.scatter(mpool.isel(unit=layer_ind, offsetsx=i), 
                    lpool.isel(unit=layer_ind, offsetsx=i),
                    color=color, s=1, alpha=0.5)
    plt.axis('square')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xlabel('max pool')
    plt.ylabel('lin pool')
    plt.title(layer)
    plt.plot([0,1], [0,1], color='r')
    
#%%
layers_to_examine = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
layers_to_examine = ['conv1', 'relu1', 'pool1', 'norm1', 'conv2', 'relu2', 'pool2',
                     'norm2', 'conv3', 'relu3', 'conv4', 'relu4', 'conv5', 'relu5',
                     'pool5', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8']

#layers_to_examine = np.unique(mpool.coords['layer_label'].values)
pool_types = [lpool,]
colors = ['r', 'g', 'b']

for layer in layers_to_examine:
    layer_ind = (mpool.coords['layer_label'] == layer).values
    #layer_ind = layer_ind & da.std(['shapes', 'shapes2', 'offsetsx'])
    plt.figure()
    
    for i, pool in enumerate(pool_types):
        plt.errorbar(2*pool.coords['offsetsx'].values[:5], 
                     pool.isel(unit=layer_ind).mean('unit')[:5], 
                     pool.isel(unit=layer_ind).std('unit')[:5], 
                     color=colors[i])
#
#    plt.axis('square')
#    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.ylabel('Correlation to linear pooling')
    plt.xlabel('Distance between stimuli (pix)')
    plt.title(layer)
#    plt.plot([0,1], [0,1], color='r')

#%%
unit = 5000
print(np.corrcoef(max_hat.isel(unit=unit, offsetsx=0).values.T.ravel(), 
                  da.isel(unit=unit, offsetsx=0).values.ravel()))
plt.figure()
max_hat.isel(unit=unit, offsetsx=0).plot()
plt.figure()
sum_hat.isel(unit=unit, offsetsx=0).plot()
plt.figure()
da.isel(unit=unit, offsetsx=0).transpose().plot()


#%%
import pandas as pd
import pickle as pk
def open_cnn_analysis(fn):
    try:
        an=pk.load(open(fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(fn,'rb'))
    fvx = an[0].sel(concat_dim='r')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    return cnn


indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
layer_label = [da.coords['layer_label'].values[index] for index in sorted(indexes)]
   
fns = [
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_null_analysis.p'
]
fns = [
'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
'blvc_caffenet_iter_1pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_null_analysis.p'
]
#    fns = [
#    'bvlc_caffenet_reference_increase_wt_cov_random0.9pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
#    'blvc_caffenet_iter_1pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
#    'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_null_analysis.p'
#    ]
results_dir = data_dir 
alt = open_cnn_analysis(results_dir +  fns[0])

k = alt['k']
init = open_cnn_analysis(results_dir + fns[1])
rm = mpool.to_pandas().transpose()
alt.index = alt.index.droplevel('layer_label')
#%%
ind1 = 1000
ind2= 2000

a = pd.concat([alt, rm], axis=1, join='inner')
plt.scatter(a['ti_in_rf'].iloc[ind1:ind2], a[16].iloc[ind1:ind2])




#%%























