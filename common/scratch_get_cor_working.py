#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:15:56 2018

@author: dean
"""



import os, sys
import numpy as np
import xarray as xr
import apc_model_fit as ac
import pandas as pd



import d_net_analysis as dn
import pickle as pk
import re

save_dir =  '/loc6tb/'
load_dir = '/loc6tb/'

model_file = load_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(model_file, chunks={'models':1000, 'shapes':370})['resp']
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
'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',
'blvc_caffenet_iter_1pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',
]

cnn_resp =['bvlc_reference_caffenetpix_width[ 8.4096606]_x_(64, 164, 51)_y_(114.0, 114.0, 1)PC370',
           'bvlc_reference_caffenetpix_width[ 8.4096606]_x_(64, 164, 51)_y_(114.0, 114.0, 1)PC370',
           'blvc_caffenet_iter_1pix_width[ 8.4096606]_x_(64, 164, 51)_y_(114.0, 114.0, 1)PC370']

fit_over_dims=None
cnn_resp_name = cnn_resp[0]
subsample_units = 100
da = xr.open_dataset(load_dir + 'data/responses/v4cnn/' + cnn_resp_name  + '.nc' )['resp']
da = da.sel(unit=slice(0, None, subsample_units)).load().squeeze()
#if null:
#    np.random.seed(1)
#    for  x in range(len(da.coords['x'])):
#        for unit in range(len(da.coords['unit'])):
#            da[1:, x, unit] = np.random.permutation(da[1:,x,unit].values)
center_pos = np.round((len(da.coords['x'])-1)/2.).astype(int)
da_0 = da.sel(x=da.coords['x'][center_pos])
#typically takes da, data, and dm, a set of linear models, an fn to write to,
#and finally fit_over_dims which says over what dims is a models fit supposed to hold.
da = da_0
#%%
da = da.reindex_like(dmod+da)
dmod = dmod.reindex_like(dmod+da)#reindex to the intersection of both

da = da - da.mean(('shapes'))
da = da.load()
ats = dmod.attrs
dmod = dmod - dmod.mean(('shapes'))

#%%
#dmod = dmod/dmod.vnorm(('shapes'))
dmod = dmod/dmod.dot(dmod, 'shapes')**0.5
#dmod = dmod/(dmod**2).sum('shapes')

#resp_n = da.vnorm(('shapes'))
resp_n = da.dot(da, 'shapes')**0.5
#resp_n = (da**2).sum(('shapes'))**0.5
#%%
proj_resp_on_model = da.dot(dmod)
#%%
if not fit_over_dims == None:
    #resp_norm = resp_n.vnorm(fit_over_dims)
    resp_norm = resp_n.dot(resp_n, fit_over_dims)**0.5
    #resp_norm = (resp_n**2).sum(fit_over_dims)**0.5

    proj_resp_on_model_var = proj_resp_on_model.sum(fit_over_dims)
    n_over = 0
    #count up how many unit vectors you'll be applying for each r.
    for dim in fit_over_dims:
        n_over = n_over + len(da.coords[dim].values)
else:
    resp_norm = resp_n
    proj_resp_on_model_var = proj_resp_on_model
    n_over = 1
#%%
all_cor = (proj_resp_on_model_var) / (resp_norm * (n_over**0.5))

#%%
all_cor = all_cor.load()
all_cor = all_cor.fillna(-666)
#%%

corarg = all_cor.argmax('models')

#%%
dmod = dmod.squeeze()
model_fit_params = dmod.coords['models'].load()[corarg]
cor = all_cor.max('models', skipna=True)
cor[cor==-666] = np.nan
for key in model_fit_params.coords.keys():
    if len(model_fit_params[key].values.shape + (1,))>1:
        cor[key] = ('unit', np.squeeze(model_fit_params[key]))

cor['models'] = ('unit', corarg.values)