# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:45:48 2016

@author: dean
"""

import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')

top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')
import d_misc as dm
import xarray as xr
import apc_model_fit as ac

def da_coef_var(da):
    #take xarray and return coefficient of variation
    #expects shapes X unit
    da = v4_resp_apc.load()
    da_min_resps = da.min('shapes')
    lessthanzero = da_min_resps<0
    if any(lessthanzero):
        da[:, da_min_resps<0] = da[:, da_min_resps<0] - da_min_resps[da_min_resps<0]
    mu = da.mean('shapes')
    sig = da.reduce(np.std, dim='shapes')
    return 1./(((mu/sig)**2)+1)

results_folder = top_dir + 'data/an_results/reference/'
cnn_name = 'APC362_scale_0.45_pos_(-7, 7, 15)_iter_0.nc'
v4_name = 'V4_362PC2001.nc'

#load v4 data
#load alex data
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name, chunks = {'shapes':370})['resp']
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
alex_resp = xr.open_dataset(top_dir + 'data/responses/' + cnn_name, chunks = {'shapes':370})['resp']
alex_resp_0 = alex_resp.sel(x=0).squeeze()

#########################
#coefficient of variation
v4_coef_var = da_coef_var(v4_resp_apc.load().copy())
alex_coef_var = da_coef_var(alex_resp_0.load().copy())

alex_coef_var.to_dataset(name='spar').to_netcdf(results_folder + 'spar_' + cnn_name)
v4_coef_var.to_dataset(name='spar').to_netcdf(results_folder + 'spar_' + v4_name)


#########################
#translation invariance
v4_resp_ti = xr.open_dataset(top_dir + 'data/an_results/v4_ti_resp.nc')['resp'].load()

def take_intersecting_1d_index(indexee, indexer):
    drop_dims = set(indexer.dims) - set(indexee.dims)
    keep_dims = set(indexee.dims) & set(indexer.dims)
    new_coords = indexer.coords.merge(indexer.coords).drop(drop_dims)
    new_dims = ([d for d in indexer.dims if d in keep_dims])

    return xr.DataArray(np.squeeze(indexee.values), new_coords, new_dims)


def translation_invariance(da):
    da = da.transpose('unit', 'x', 'shapes')
    
    da_ms = (da - da.mean(['shapes'])).squeeze()
    no_na = [unit.dropna('shapes', how='all').dropna('x', how='all') for unit in da_ms ]
    s = [np.linalg.svd(unit.transpose('shapes', 'x').values, compute_uv=0) for unit in no_na]
    best_r_alex = np.array([(asingval[0]**2)/(sum(asingval**2)) for asingval in s])
    
    ti = xr.DataArray(np.squeeze(best_r_alex), dims='unit')
    ti = take_intersecting_1d_index(ti, da)
    
    return ti
    
#ti_v4 = translation_invariance(v4_resp_ti)
#alex_resp = alex_resp.load().squeeze()[:, :, :]
#ti_alex = translation_invariance(alex_resp)

############
#APC measurement
import pickle
with open(top_dir + 'data/models/PC370_params.p', 'rb') as f:
    shape_dict_list = pickle.load(f)
shape_id = v4_resp_apc.coords['shapes'].values
shape_dict_list = [shape_dict_list[sn] for sn in shape_id.astype(int)]

maxAngSD = np.deg2rad(171); minAngSD = np.deg2rad(23)
maxCurSD = 0.98; minCurSD = 0.09;
nMeans = 16; nSD = 16
fn = top_dir + 'data/models/' + 'apc_models_362_16x12.nc'
dam = ac.make_apc_models(shape_dict_list, shape_id, fn, nMeans, nSD,
                         maxAngSD, minAngSD, maxCurSD, minCurSD,
                         prov_commit=False, save=True, replace_prev_model=False)['resp']

dam_n = dam.copy()
#shuffle columns
_ = dam_n.values
for ind in range(_.shape[1]):
    np.random.shuffle(_[:,ind])

null_cor_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'unit':100, 'shapes':370}), 
                                     dam_n.chunk({'models':1000, 'shapes':370}), 
                                    fit_over_dims=None, prov_commit=False)    
alt_cor_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'unit':100, 'shapes':370}), 
                                     dam.chunk({'models':1000, 'shapes':370}), 
                                    fit_over_dims=None, prov_commit=False)  


    