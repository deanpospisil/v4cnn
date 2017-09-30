#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:27:25 2017

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

def open_cnn_analysis(fn, layer_label):
    try:
        an=pk.load(open(fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(fn,'rb'))
    fvx = an[0].sel(concat_dim='r')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    return fvx, rf, cnn

data_dir = '/loc6tb/'
cnn_names =['bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',]
da = xr.open_dataset(data_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
layer_label = [da.coords['layer_label'].values[index] 
                for index in sorted(indexes)]
   
fn = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p'
results_dir = data_dir + 'data/an_results/'
alt = open_cnn_analysis(results_dir +  fn, layer_label)[-1]

#lets get all indices of units with preference for shapes with preference to left

center = np.deg2rad(90.)
width = np.deg2rad(100)

inds = (((alt['apc']**0.5)>0.5) & (alt['cur_mean']>0.9)
        & (alt['or_mean']>(center-width)) & (alt['or_mean']<(center+width)))
print('there are '+ str(inds.sum()) + ' units')

good_units = alt[inds]
good_inds = good_units.index.get_level_values(1).values


fn = 'bvlc_reference_caffenet_pix_width[64.0]_x_(114.0, 114.0, 1)_y_(114.0, 114.0, 1)_rots_100_angPosTest'
da = xr.open_dataset(data_dir + 'data/responses/' + fn + '.nc')['resp']
da = da.squeeze()
da_g = da[..., good_inds] 
da_g_p= da_g[:]
da_g_p = da_g_p - da_g_p.mean(['shapes', 'rotation'])
da_g_p = da_g_p/((da_g_p**2).sum(['shapes', 'rotation']))**0.5

plt.imshow(da_g_p[0].values.T)
plt.figure()
((da_g_p)**2).mean(['unit'])[0].plot()
#%%
var_ang_pos = da_g[9:, 5].var('shapes')
var_point_ori = da_g[:9, 5].var('shapes')

more_ang_pos_var = var_ang_pos/(var_point_ori + var_ang_pos)

plt.hist(more_ang_pos_var.values[~np.isnan(more_ang_pos_var.values)], range=(0,1))

plt.figure()
ind = int(more_ang_pos_var.argsort()[2].values)
da_g[..., ind].plot()
print(more_ang_pos_var[ind])