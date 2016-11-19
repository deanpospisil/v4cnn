# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:06:55 2016

@author: dean
"""
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')
import matplotlib
from matplotlib.ticker import FuncFormatter

import xarray as xr
import apc_model_fit as ac
import pandas as pd
import matplotlib.ticker as mtick
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except:
    print('no plot')

import pickle
import d_net_analysis as dn
import pickle as pk
import re
save_dir = '/dean_temp/'
load_dir = '/dean_temp/'
save_dir = top_dir
load_dir = top_dir

measure_list =[ 'apc', 'ti', 'ti_orf', 'cv_ti', 'k', 'in_rf', 'no_response_mod']
measure_list = ['k', 'ti_av_cov']
#measure_list = ['k', 'ti_av_cov']

model_file = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(model_file, chunks={'models':50, 'shapes':370})['resp']
cnn_resp =[
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
#'bvlc_reference_caffenetAPC362_pix_width[64.0]_pos_(64.0, 164.0, 51)',
#'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
#'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[64.0]_pos_(64.0, 164.0, 51)',
'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
]
null = False
#w = 32
subsample_units = 1

for cnn_resp_name in cnn_resp:
    w = int(float( re.findall('\[\d\d.0', cnn_resp_name)[0][1:]))
    da = xr.open_dataset(load_dir + 'data/responses/' + cnn_resp_name  + '.nc' )['resp']
    da = da.sel(unit=slice(0, None, subsample_units)).load().squeeze()
    if null:
        np.random.seed(1)
        for  x in range(len(da.coords['x'])):
            for unit in range(len(da.coords['unit'])):
                da[1:, x, unit] = np.random.permutation(da[1:,x,unit].values)
    center_pos = np.round(len(da.coords['x'])/2.).astype(int)
    da_0 = da.sel(x=da.coords['x'][center_pos])
    rf = dn.in_rf(da, w=w)
    measures = []
    if 'apc' in measure_list:	
        measures.append(dn.ac.cor_resp_to_model(da_0.chunk({'shapes': 370}),
                                                dmod, fit_over_dims=None,
                                                prov_commit=False).values**2)
    if 'ti' in measure_list:
        measures.append(dn.SVD_TI(da, rf))
    if 'ti_orf' in measure_list:
        measures.append(dn.SVD_TI(da))
    if 'cv_ti' in measure_list:
        measures.append(dn.cross_val_SVD_TI(da, rf))
    if 'k' in measure_list:		
        measures.append(list(dn.kurtosis(da_0)))
    if 'in_rf' in measure_list:
        measures.append(np.sum(rf,1))
    if 'no_response_mod' in measure_list:
        measures.append((((da-da.mean('shapes'))**2).sum(['shapes','x'])==0).values)
    if 'ti_av_cov':
        measures.append(dn.ti_av_cov(da, rf))

    keys = ['layer_label', 'unit']
    coord = [da_0.coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    pda = pd.DataFrame(np.array(measures).T, index=index, columns=measure_list)
    receptive_field = (da.drop(-1,dim='shapes')**2).sum('shapes')**0.5
    da_cor = da.copy()
    da_cor -= da_cor.mean('shapes')
    da_cor /= da_cor.chunk({}).vnorm('shapes') 
    da_cor_0 = da_cor.isel(x=center_pos) 
    correlation = (da_cor_0*da_cor).sum('shapes')**2
    pos_props = xr.concat([correlation,receptive_field],dim=['r2','rf']) 
    all_props = [pos_props, pda]
    if null:
        pk.dump(all_props, open(save_dir  + 'data/an_results/' + cnn_resp_name  + '_null_analysis.p','wb'))
    else:
        pk.dump(all_props, open(save_dir  + 'data/an_results/' + cnn_resp_name  + '_analysis_home.p','wb'))
#    pk.load(open(top_dir + 'data/an_results/' + cnn_resp_name  + '_analysis.p','rb'))
