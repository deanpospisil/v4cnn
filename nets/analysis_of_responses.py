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
measure_list =[ 'apc', 'ti', 'ti_orf', 'cv_ti', 'k', 'in_rf', 'no_response_mod']
measure_list = ['apc','ti', 'ti_orf', 'k', 'ti_av_cov']
model_file = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(model_file, chunks={'models':50, 'shapes':370})['resp']
cnn_resp =['bvlc_reference_caffenetAPC362_pix_width[30.0]_pos_(64.0, 164.0, 101)',]
null = False
w = 30
subsample_units = 100

for cnn_resp_name in cnn_resp:
    da = xr.open_dataset(top_dir + 'data/responses/' + cnn_resp_name  + '.nc' )['resp']
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
        measures.append(list(dn.kurtosis(da_0.drop(-1, dim='shapes')).values))
    if 'in_rf' in measure_list:
        measures.append(np.sum(rf,1))
    if 'no_response_mod' in measure_list:
        measures.append((((da-da.mean('shapes'))**2).sum(['shapes','x'])==0).values)
    if 'ti_av_cov':
        measures.append(dn.ti_av_cov(da, None))

    keys = ['layer_label', 'unit']
    coord = [da_0.coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    pda = pd.DataFrame(np.array(measures).T, index=index, columns=measure_list)
#    pda.to_pickle(top_dir + 'data/an_results/' + cnn_resp_name  + '_analysis.p')
    receptive_field = (da.drop(-1,dim='shapes')**2).sum('shapes')**0.5
    da_cor = da.copy()
    da_cor -= da_cor.mean('shapes')
    da_cor /= da_cor.chunk({}).vnorm('shapes')
    da_cor_0 = da_cor.isel(x=center_pos) 
    correlation = (da_cor_0*da_cor).sum('shapes')**2
    pos_props = xr.concat([correlation,receptive_field],dim=['r2','rf']) 
    all_props = [pos_props, pda]
    pk.dump(all_props, open(top_dir + 'data/an_results/' + cnn_resp_name  + '_analysis.p','wb'))
    pk.load(open(top_dir + 'data/an_results/' + cnn_resp_name  + '_analysis.p','rb'))
'''
unit_resp = da.sel(unit=8990).drop(-1, dim='shapes').transpose('x', 'shapes').values
unit_resp -= unit_resp.mean(1,keepdims=True)
cov = np.dot(unit_resp, unit_resp.T)
cov[np.diag_indices_from(cov)] = 0
numerator = np.sum(np.triu(cov))
vlength = np.linalg.norm(unit_resp, axis=1)
max_cov = np.outer(vlength.T, vlength)
max_cov[np.diag_indices_from(max_cov)] = 0
denominator= np.sum(np.triu(max_cov))
frac_var = numerator/denominator
print(frac_var)
print(pda['ti_av_cov'].argmax())
print(pda['ti_av_cov'].max())
'''
