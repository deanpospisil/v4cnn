# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:34:09 2017

@author: deanpospisil
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
import apc_model_fit as ac
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp
import d_net_analysis as dn
import caffe_net_response as cf


def open_cnn_analysis(fn, layer_label):
    try:
        an=pk.load(open(fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(fn, 'rb'))
    fvx = an[0].sel(concat_dim='r')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    return fvx, rf, cnn

cnn_name = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)'
data_dir = '/Users/deanpospisil/Desktop/modules/v4cnn/'
da = xr.open_dataset(data_dir + 'data/responses/' + cnn_name + '.nc')['resp']
da = da.sel(unit=slice(0, None, 1)).squeeze()
middle = np.round(len(da.coords['x'])/2.).astype(int)
da_0 = da.sel(x=da.coords['x'][middle])

indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
layer_label = [da.coords['layer_label'].values[index] for index in sorted(indexes)]
   
fn = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p'

#    fns = [
#    'bvlc_caffenet_reference_increase_wt_cov_random0.9pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
#    'blvc_caffenet_iter_1pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
#    'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_null_analysis.p'
#    ]
results_dir = data_dir + 'data/an_results/'
alt = pd.concat([open_cnn_analysis(results_dir +  fn, layer_label)[-1], alt_v4], axis=0)
