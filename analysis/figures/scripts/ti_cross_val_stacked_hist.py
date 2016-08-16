# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:34:45 2016

@author: dean
"""

import numpy as  np
import scipy.io as  l
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as mtick
#
import matplotlib as mpl
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm
import pickle as pk
import pandas as pd
def tick_format_d(x, pos):
    if x==0:
        return('0')
    else:
        if x>=1:
            return(str(x).split('.')[0])
        else:
            return(np.round(x,2))

def nice_axes(axes, xticks=None, yticks=None, nxticks=5, nyticks=2):

    for i, an_axes in enumerate(axes):
        if i==len(axes)-1:
            if yticks==None:
                an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
            else:
                an_axes.set_yticks(yticks)
            if xticks==None:
               an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
            else:
                an_axes.set_xticks(xticks)
                an_axes.xaxis.set_tick_params(length=0)
                an_axes.yaxis.set_tick_params(length=0)
                an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
            an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
        else:
            an_axes.set_xticks([])
            an_axes.set_yticks([])

def stacked_hist_layers(cnn, logx=False, logy=False, xlim=None, maxlim=False, bins=100):
    layers = cnn.index.get_level_values('layer_label').unique()
    if logx:
        cnn = np.log(cnn.dropna())
    if maxlim:
        xlim = [np.min(cnn.dropna().values), np.max(cnn.dropna().values)]
    for i, layer in enumerate(layers):
        plt.subplot(len(layers), 1, i+1)
        vals = cnn.loc[layer].dropna().values.flatten()


        plt.hist(vals, log=logy, bins=bins, histtype='step',
                 range=xlim, normed=False)

        plt.plot([np.median(vals),]*2, np.array(plt.gca().get_ylim()), color='red')
        plt.gca().set_ylabel(layer, ha='right', rotation=0, labelpad=25)
        plt.gca().yaxis.set_label_position("right")

    if logx:
        plt.xlabel('log')
    nice_axes(plt.gcf().axes)
    #plt.suptitle(cnn.name)
def take_intersecting_1d_index(indexee, indexer):
    drop_dims = set(indexer.dims) - set(indexee.dims)
    keep_dims = set(indexee.dims) & set(indexer.dims)
    new_coords = indexer.coords.merge(indexer.coords).drop(drop_dims)
    new_dims = ([d for d in indexer.dims if d in keep_dims])

    return xr.DataArray(np.squeeze(indexee.values), new_coords, new_dims)

def translation_invariance(da):
    da = da.transpose('unit', 'x', 'shapes')

    da_ms = (da - da.mean(['shapes'])).squeeze()
    if da_ms.isnull().sum()>0:
    	no_na = [unit.dropna('shapes', how='all').dropna('x', how='all') for unit in da_ms ]
    else:
	no_na = [unit for unit in da_ms]
    s = [np.linalg.svd(unit.transpose('shapes', 'x').values, compute_uv=0) for unit in no_na]
    best_r = np.array([(asingval[0]**2)/(sum(asingval**2)) for asingval in s])

    ti = xr.DataArray(np.squeeze(best_r), dims='unit')
    ti = take_intersecting_1d_index(ti, da)

    return ti
def kurtosis(da):
    #take xarray and return coefficient of variation
    #expects shapes X unit
    da = da.transpose('shapes', 'unit')
    mu = da.mean('shapes')
   # k = da.reduce(kurtosis,dim='shapes')
    sig = da.reduce(np.var, dim='shapes')
    k = (((da - mu)**4).sum('shapes')/da.shapes.shape[0])/(sig**2)
    return k

cnn_name = 'APC362_scale_1_pos_(-99, 96, 66)bvlc_reference_caffenet'
alex_resp = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp'].squeeze().load()
#cnn_name = 'APC362_scale_1_pos_(-7, 7, 15)_ref_iter_0.nc'
#alex_resp = xr.open_dataset(top_dir + 'data/responses/' + cnn_name )['resp'].squeeze().load()

ti_alex = translation_invariance(alex_resp)
alex_resp = alex_resp.sel(x=0)


drop = ['conv4_conv4_0_split_0', 'conv4_conv4_0_split_1']
for drop_name in drop:
    alex_resp = alex_resp[:,(alex_resp.coords['layer_label'] != drop_name)]
    ti_alex = ti_alex[(ti_alex.coords['layer_label'] != drop_name)]


k_alex = kurtosis(alex_resp)
# get index for pandas
keys = [key for key in alex_resp['unit'].coords.keys()
        if not alex_resp['unit'].coords[key].values.shape==() and key!='unit']
keys = ['layer_label', 'layer_unit']
coord = [alex_resp['unit'].coords[key].values
        for key in keys]
index = pd.MultiIndex.from_arrays(coord, names=keys)

fn = top_dir + 'data/an_results/ti_cross_val_' + cnn_name
ticv_alex = pk.load(open(fn, 'rb'))
ticv_alex = pd.DataFrame(ticv_alex.values, index=index)
ti_alex = pd.DataFrame(ti_alex.values, index=index)

plt.figure()
stacked_hist_layers(ticv_alex[k_alex.values<40].dropna(), logx=False, logy=False,  xlim=[0,1] )
plt.suptitle(cnn_name +'  steps of 3, widest stimuli width of 24 pixels')
plt.xlabel('Average R^2 for cross-validation')
plt.savefig(top_dir + 'analysis/figures/images/cv_translation_invariance_by_layer '+ cnn_name + '.png')


plt.figure()
stacked_hist_layers(ti_alex[k_alex.values<40], logx=False, logy=False,  xlim=[0,1])
plt.suptitle(cnn_name +'  steps of 3, widest stimuli width of 24 pixels')
plt.xlabel('Average R^2 for cross-validation')
plt.savefig(top_dir + 'analysis/figures/images/normal_translation_invariance_by_layer '+ cnn_name + '.png')


plt.figure()
plt.scatter(ticv_alex[k_alex.values<40], ti_alex[k_alex.values<40], alpha=0.1, s=1)
plt.xlabel('TI by SVD leave one out cross validation')
plt.ylabel('TI by SVD')
plt.plot([0,1],[0,1], color='red')
plt.savefig(top_dir + 'analysis/figures/images/comparison_cv_normal_translation_invariance_by_layer '+ cnn_name + '.png')
