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
    da_min_resps = da.min('shapes')
    da[:, da_min_resps<0] = da[:, da_min_resps<0] - da_min_resps[da_min_resps<0]
    mu = da.mean('shapes')
    sig = da.reduce(np.std, dim='shapes')
    return 1./(((mu/sig)**2)+1)

results_folder = top_dir + 'data/an_results/reference/'
cnn_name = 'APC362_scale_1_pos_(-7, 7, 15)_ref_iter_0.nc'
v4_name = 'V4_362PC2001.nc'
#load v4 data
#load alex data
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name, chunks = {'shapes':370})['resp']
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

#def translation_invariance(da)      :

da = v4_resp_ti
da = alex_resp.load().squeeze()
da = da.transpose('unit', 'x', 'shapes')
da_ms = (da - da.mean(['shapes'])).squeeze()
no_na = [unit.dropna('shapes', how='all').dropna('x', how='all') for unit in da_ms ]
s = [np.linalg.svd(unit.transpose('shapes', 'x').values, compute_uv=0) for unit in no_na]
best_ti = np.array([(asingval[0]**2)/(sum(asingval**2)) for asingval in s])

ti = xr.DataArray(np.squeeze(best_ti), dims='unit')
ti = take_intersecting_1d_index(ti, da)

#need to transmit all the meta-data, turn this into a definition.

#translation_invariance(v4_resp_ti)
##ti.attrs['resp_coords'] = da_ms.coords.values
#ti.to_dataset(name='tin').to_netcdf(ti_name)
#t=da[:,:,1]
#s=np.linalg.svd(t.values, compute_uv=0)
#print(s)
#s=np.linalg.svd(t.values.T, compute_uv=0)
#print(s)
