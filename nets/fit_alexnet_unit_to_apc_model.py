#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:31:21 2018

@author: dean
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.insert(0, top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr;import pandas as pd
import apc_model_fit as ac
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp
import d_net_analysis as dn
from sklearn.neighbors import KernelDensity
import caffe_net_response as cf
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

#%%   
#v4 fit to CNN and APC 
v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
file = open(top_dir + 'data/responses/v4_apc_109_neural_labels.txt', 'r')
wyeth_labels = [label.split(' ')[-1] for label in 
            file.read().split('\n') if len(label)>0]
v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
fn = top_dir + 'data/models/' + 'apc_models_362.nc'

if 'apc_fit_v4' not in locals():
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()
    apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                      dmod.chunk({}), 
                                      fit_over_dims=None, 
                                      prov_commit=False)**2

cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
            'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)']
colors = ['r','g','b','m','c', 'k', '0.5']
from sklearn.model_selection import ShuffleSplit
X = np.arange(362)
cv_scores = []
model_ind_lists = []
models = []
for cnn_name in cnn_names:
    da = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp']
    da = da.sel(unit=slice(0, None, 1)).squeeze()
    middle = np.round(len(da.coords['x'])/2.).astype(int)
    da_0 = da.sel(x=da.coords['x'][middle])
    da_0 = da_0.sel(shapes=v4_resp_apc.coords['shapes'].values)
    models.append(da_0)
models.append(dmod)

