#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:43:19 2017

@author: dean
"""
import numpy as np
import pandas as pd
import pickle as pk
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.insert(0, top_dir + 'xarray/');
import xarray as xr

def open_cnn_analysis(fn):
    try:
        an=pk.load(open(fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(fn,'rb'))
    cnn = an[1]
    return cnn

data_dir = '/loc6tb/'
fn  ='bvlc_reference_caffenetpix_width[32.0]_x_(113.0, 113.0, 1)_y_(113.0, 113.0, 1)_offsets_PC370'
#fn = 'blvc_caffenet_iter_1pix_width[32.0]_x_(113.0, 113.0, 1)_y_(113.0, 113.0, 1)_offsets_PC370'
da = xr.open_dataset(data_dir + 'data/responses/' + fn + '.nc')['resp'].squeeze()

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

fns[0] = 'bvlc_reference_caffenetpix_width[100.0]_x_(64, 164, 52)_y_(114.0, 114.0, 1)PC370_analysis.p'
results_dir = data_dir + 'data/an_results/'
alt = open_cnn_analysis(results_dir +  fns[0])
cn = alt
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
for layer in layers:
    cn_sub = cn.loc[layer]
    small_orsd = cn_sub['or_sd']<1
    good_apc = (cn_sub['apc']**0.5)>0.6
    good_ti = cn_sub['ti_in_rf']>0.7
    point = cn_sub['cur_mean']>0.5
    
    all_crit = small_orsd & good_apc & good_ti & point
     
    #print(np.sum(all_crit))
    
    
    c_b = cn_sub[['apc', 'ti_in_rf', 'cur_mean', 'cur_sd', 'or_mean', 'or_sd']].loc[all_crit]
    
    c_b['apc'] = c_b['apc']**0.5
    
    c_b['or_sd'] = np.rad2deg(c_b['or_sd'])
    c_b['or_mean'] = np.rad2deg(c_b['or_mean'])
    c_b = np.round(c_b, 3)
    print(c_b)
    
#c_b.to_csv(top_dir+ '/v4cnn/data/an_results/best_red_large.csv')

# %% 
fn = 'bvlc_reference_caffenetpix_width[100.0]_x_(64, 164, 52)_y_(114.0, 114.0, 1)PC370.nc'
da = xr.open_dataset(data_dir + 'data/responses/' + fn)['resp'].squeeze()
da = da.isel(x=26)
sd = da.var('shapes')**0.5
mu = da.mean('shapes')


p_dyn = sd + mu

layer = 'fc'
s = alt.drop('v4', level='layer_label').loc[layer].drop(['ti_av_cov', 'k', 'k_stim', 'k_pos', 'models'], axis=1)
s['apc'] = s['apc']**0.5
p_dyn_s = p_dyn[p_dyn.coords['layer_label'] == layer]
p_dyn_s = p_dyn_s.to_pandas()

all_vals = np.concatenate([s.values, p_dyn_s[:,np.newaxis]], 1)
all_pd = pd.DataFrame(all_vals, columns=list(s.columns.values) + ['dyn',])
from pandas.plotting import scatter_matrix
scatter_matrix(all_pd, alpha=0.1, figsize=(10,10))
all_pd['or_sd'] = np.rad2deg(all_pd['or_sd'])
all_pd['or_mean'] = np.rad2deg(all_pd['or_mean'])

#%%
small_orsd = all_pd['or_sd']<48
good_apc = (all_pd['apc'])>0.7
good_ti = all_pd['ti_in_rf']>0.7
point = all_pd['cur_mean']>0.5
dyn = all_pd['dyn']>5


all_crit = small_orsd & good_apc & good_ti & point & dyn
rb = all_pd.loc[all_crit]
print(np.sum(all_crit))

#rb.to_csv(top_dir+ '/v4cnn/data/an_results/best_red_large_large_dyn.csv')