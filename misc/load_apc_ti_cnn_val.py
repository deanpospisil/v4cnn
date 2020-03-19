
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:15:34 2017

@author: deanpospisil
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:14:18 2016

@author: dean
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
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


def open_cnn_analysis(fn, layer_label):
    try:
        an=pk.load(open(top_dir + 'data/an_results/' + fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(top_dir + 'data/an_results/' + fn,'rb'))
    fvx = an[0].sel(concat_dim='r2')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    return fvx, rf, cnn

def process_V4(v4_resp_apc, v4_resp_ti, dmod):
    ti = dn.ti_av_cov(v4_resp_ti, rf=None)
    apc = dn.ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                  dmod.chunk({}), fit_over_dims=None, 
                                    prov_commit=False)**2.
    k_apc = list(dn.kurtosis(v4_resp_apc).values)
    k_ti = list(dn.kurtosis(v4_resp_ti.mean('x')).values)

    keys = ['layer_label', 'unit']
    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(ti)),
                                       np.arange(len(ti))], names=keys)
    v4pdti  = pd.DataFrame(np.array([ti, k_ti]).T, index=index, 
                           columns=['ti_av_cov', 'k'])

    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(apc)), 
                                       np.arange(len(apc))], 
                                       names=keys)
    coords_to_take= ['cur_mean', 'cur_sd', 'or_mean', 'or_sd', 'models']
    apc_coords = [apc.coords[coord].values for coord in coords_to_take]
    v4pdapc  = pd.DataFrame(np.array([apc.values,] + apc_coords + [k_apc,]).T, 
               index=index, 
               columns=['apc', ] + coords_to_take + [ 'k',])
    v4 = pd.concat([v4pdti, v4pdapc])
    return v4

#%%
fns = [
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_null_analysis.p'
]
cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',]
da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
da = da.sel(unit=slice(0, None, 1)).squeeze()
middle = np.round(len(da.coords['x'])/2.).astype(int)
da_0 = da.sel(x=da.coords['x'][middle])
indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
layer_label = [da.coords['layer_label'].values[index] for index in sorted(indexes)]

do = open_cnn_analysis(fns[0], layer_label)[-1]
dok = do[do['k']<40]
dok[['apc', ]] = dok[['apc',] ]**0.5
do = dok[['apc', 'ti_av_cov', 'cur_mean', 'cur_sd', 'or_mean', 'or_sd']]
layer_label_examine = [str(name).split('b')[1][1:-1] for name in layer_label][4:-1]

dist = np.sqrt((1-do['apc'])**2 +  (1-do['ti_av_cov'])**2  )  
do = pd.concat([do, dist], axis=1)
do.columns = ['apc', 'ti_av_cov', 'cur_mean', 'cur_sd', 'or_mean', 'or_sd', 'v4ness']

sort_do = [do.loc[name].sort('v4ness')[:10] for name in layer_label_examine]
df_list = []
for i, name in enumerate(layer_label_examine):
    keys = ['layer_label', 'unit']
    index = pd.MultiIndex.from_arrays([np.array([name,]*len(sort_do[i])),
                                       sort_do[i].index.values], names=keys)
    temp_df = pd.DataFrame(sort_do[i].values, 
               index=index, 
               columns=do.columns)
    df_list.append(temp_df)
top_10 = pd.concat(df_list)

rf = open_cnn_analysis(fns[0], layer_label)[1]
cor = open_cnn_analysis(fns[0], layer_label)[0]
av_cors = cor.groupby('layer_label').mean('unit')
av_rfs = rf.groupby('layer_label').mean('unit')
top_10.to_csv(top_dir + '/data/an_results/top_10.csv')

v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
file = open(top_dir + 'data/responses/v4_apc_109_neural_labels.txt', 'r')
wyeth_labels = [label.split(' ')[-1] for label in 
            file.read().split('\n') if len(label)>0]
v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()

apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                  dmod.chunk({}), 
                                  fit_over_dims=None, 
                                  prov_commit=False)

v4_resp_apc = v4_resp_apc - v4_resp_apc.mean('shapes')
v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()
alt_v4 = process_V4(v4_resp_apc, v4_resp_ti, dmod)
