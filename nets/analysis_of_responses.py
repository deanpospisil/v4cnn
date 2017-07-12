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

import xarray as xr
import apc_model_fit as ac
import pandas as pd



import d_net_analysis as dn
import pickle as pk
import re

save_dir = top_dir
load_dir = top_dir


model_file = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(model_file, chunks={'models':50, 'shapes':370})['resp']
#cnn_resp =[
#'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
#'bvlc_reference_caffenetAPC362_pix_width[64.0]_pos_(64.0, 164.0, 51)',
##'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
##'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[64.0]_pos_(64.0, 164.0, 51)',
#'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
#'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
#]

cnn_resp =[
'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',
'blvc_caffenet_iter_1pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',
'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',
]
nulls = [0, 0, 1]
subsample_units = 1

for cnn_resp_name, null  in zip(cnn_resp, nulls):
    print(cnn_resp_name)
    measure_list = ['apc', 'k_pos', 'k_stim', 'ti_in_rf']    

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
    #rf = dn.in_rf(da, w=w)
    measures = []
    if 'apc' in measure_list:	
        fit = ac.cor_resp_to_model(da_0.chunk({'shapes': 370}),
                                                dmod, fit_over_dims=None,
                                                prov_commit=False)**2
        cds = fit.coords
        #measure_list = ['k_pos', 'k_stim', 'ti_in_rf', 'apc', 'cur_mean', 'cur_sd', 'or_mean', 'or_sd', 'models'] 
        measure_list =['apc', 'cur_mean', 'cur_sd', 'or_mean', 'or_sd', 'models' ] +list(set(measure_list)-set(['apc',]))
        measures = [ fit.values, cds['cur_mean'].values, cds['cur_sd'].values, cds['or_mean'].values, cds['or_sd'].values, cds['models'].values]
    if 'ti' in measure_list:
        measures.append(dn.SVD_TI(da, rf))
    if 'ti_orf' in measure_list:
        measures.append(dn.SVD_TI(da))
    if 'cv_ti' in measure_list:
        measures.append(dn.cross_val_SVD_TI(da, rf))
    if 'k' in measure_list:		
        measures.append(list(dn.kurtosis(da_0).values))
        
    if 'in_rf' in measure_list:
        measures.append(np.sum(rf, 1))
    if 'no_response_mod' in measure_list:
        measures.append((((da-da.mean('shapes'))**2).sum(['shapes','x'])==0).values)
    if 'ti_av_cov' in measure_list:
        measures.append(dn.ti_av_cov(da, rf))
    if 'ti_in_rf' in measure_list:
        measures.append(dn.ti_in_rf(da, w))
    if 'k_stim' in measure_list and 'k_pos' in measure_list:
        k_pos, k_stim = dn.kurtosis_da(da)
        measures.append(k_pos)
        measures.append(k_stim)

    keys = ['layer_label', 'unit']
    coord = [da_0.coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    pda = pd.DataFrame(np.array(measures).T, index=index, columns=measure_list)
    receptive_field = (da.drop(-1,dim='shapes')**2).sum('shapes')**0.5
    da_cor = da.copy()
    da_cor -= da_cor.mean('shapes')
    da_cor /= (da_cor**2).sum('shapes') 
    da_cor_0 = da_cor.isel(x=center_pos) 
    correlation = (da_cor_0*da_cor).sum('shapes')
    pos_props = xr.concat([correlation,receptive_field],dim=['r', 'rf']) 
    all_props = [pos_props, pda]
    if null:
        pk.dump(all_props, open(save_dir  + 'data/an_results/' + cnn_resp_name  + '_null_analysis.p','wb'))
    else:
        pk.dump(all_props, open(save_dir  + 'data/an_results/' + cnn_resp_name  + '_analysis.p','wb'))


