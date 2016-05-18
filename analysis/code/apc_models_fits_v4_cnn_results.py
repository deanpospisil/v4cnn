# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:35:38 2016

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:23:36 2016

plots of v4 APC params

@author: dean
"""


import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
import xarray as xr

import pickle
import apc_model_fit as ac

def degen(daa):
    minfracvar = 0.5
    _ = (daa**2)
    tot_var = _.sum('shapes')
    non_zero = tot_var<1e-8
    just_one_shape = (_.max('shapes')/tot_var)>minfracvar
    degen_inds = just_one_shape + non_zero
    return degen_inds

def apc_model_cors_and_nulls(ds, dmod):
    ds_list = []
    dmod = dmod.chunk({'models':1000, 'shapes':370})
    for key in ds.keys():
        daa = ds[key].load()
        degen_inds = degen(daa)
        daa = daa[:,-degen_inds]

        cor = ac.cor_resp_to_model(daa.chunk({'unit':100, 'shapes':370}),
                                   dmod, fit_over_dims=None, prov_commit=False)

        shape_ind = range(daa['shapes'].count().values)

        for i in range(daa['unit'].count().values):
            np.random.shuffle(shape_ind)
            daa[:, i] = daa[shape_ind, i]

        cor_shuf = ac.cor_resp_to_model(daa.chunk({'unit':100, 'shapes':370}),
                           dmod, fit_over_dims=None, prov_commit=False)
        dict_ds = {'real':cor ,'shuf':cor_shuf }
        ds_list.append(dict_ds)
    return ds_list

#open those responses, and build apc models for their shapes
with open(top_dir + 'data/models/PC370_params.p', 'rb') as f:
    shape_dict_list = pickle.load(f)

da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
daa=daa.loc[:, 0, :]#without translation
daa.attrs['type'] = 'AlexNet'
da.attrs['type'] = 'V4'
shape_id = da.coords['shapes'].values
shape_dict_list = [shape_dict_list[sn] for sn in shape_id.astype(int)]
maxAngSD = np.deg2rad(171); minAngSD = np.deg2rad(23)
maxCurSD = 0.98; minCurSD = 0.09;
maxCurSD = 0.98; minCurSD = 0.01
nMeans = 16; nSD = 16
fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dam = ac.make_apc_models(shape_dict_list, shape_id, fn, nMeans, nSD,
                         maxAngSD, minAngSD, maxCurSD, minCurSD,
                         prov_commit=False, save=True, replace_prev_model=True)

#load the models you made, and fit them to the cells responses
dmod = xr.open_dataset(fn, chunks={'models': 1000, 'shapes': 370}  )['resp']
ds = {'v4':da, 'cnn':daa}

ds_list = apc_model_cors_and_nulls(ds, dmod)
with open(top_dir + 'data/models/ds_list.p','wb') as f:
    pickle.dump(ds_list, f)

with open(top_dir + 'data/models/ds_list.p', 'rb') as f:
    d_rec = pickle.load(f)

