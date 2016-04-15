# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:39:42 2016

@author: dean
"""
import os, sys
import numpy as np
import warnings
import pickle
import matplotlib.pyplot as plt

top_dir = os.getcwd().split('net_code')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir +'common')

import d_misc as dm
import xarray as xr
import apc_model_fit as ac

quick = False

#open those responses, and build apc models for their shapes
with open(top_dir + 'data/models/PC370_params.p', 'rb') as f:
    shape_dict_list = pickle.load(f)

a = np.hstack((range(14), range(18,318)))
a = np.hstack((a, range(322, 370)))
shape_id = a

maxAngSD = np.deg2rad(171)
minAngSD = np.deg2rad(23)
maxCurSD = 0.98
minCurSD = 0.09
nMeans = 16
nSD = 16
fn = 'apc_models.nc'
dmod = ac.make_apc_models(shape_dict_list, range(370), fn, nMeans, nSD, maxAngSD, minAngSD,
                      maxCurSD, minCurSD, prov_commit=True)['resp']

dmod = dmod.drop(set(range(370)) - set(shape_id), dim='shapes').chunk({'models':100})

all_iter = dm.list_files(top_dir + 'data/responses/iter_*.nc')
it_num = []
for fn in all_iter:
    it_num.append( int(fn.split('iter_')[1].split('.')[0] ))
all_iter = [all_iter[ind] for ind in  np.argsort(it_num)]
all_iter = [all_iter[ind.astype(int)] for ind in dm.grad_aprx_ind(len(it_num))]
#da_c = xr.open_dataset(all_iter[0], chunks = {'unit':100, 'shapes': 370})['resp']


for fn in all_iter:
    da_c = xr.open_dataset(fn, chunks = {'unit':100})['resp']
    da_c = da_c.drop(set(range(370)) - set(shape_id) , dim='shapes')

    if quick:

        dmod = dmod.sel(models=range(10), method='nearest')
        da_c = da_c.sel(unit=range(30),  method='nearest')
        da_c = da_c.sel(x=[0, 2],  method='nearest')
    print(fn)
    fn = top_dir + 'data/an_results/r_apc_models_unique' + str(fn.split('iter')[1])
    if not os.path.isfile(fn):
        cor = ac.cor_resp_to_model(da_c, dmod, fit_over_dims = ('x',), prov_commit=True)
        cor.to_dataset(name='r').to_netcdf(fn)
    else:
        print('already written')
        warnings.warn('Fit  File has Already Been Written.')
        cor = xr.open_dataset(fn)

    #cor = ac.cor_resp_to_model(da_c, dmod, fit_over_dims = ('x',))
    #cor.to_dataset(name='r').to_netcdf(top_dir + 'data/an_results/r_apc_models_unique' + str(fn.split('iter')[1]) )



#ds = xr.open_mfdataset(top_dir + 'data/an_results/r_over_train/r_*.nc', concat_dim = 'niter')
all_iter = dm.list_files(top_dir + 'data/an_results/r_over_train/r_*.nc')
it_num = []
for fn in all_iter:
    it_num.append( int(fn.split('e_')[1].split('.')[0] ))
all_iter = [all_iter[ind] for ind in  np.argsort(it_num)]
  
n_v4 = []
for fn in all_iter:
     d = (xr.open_dataset(fn)['r']>0.5)
     d = d.loc[d.coords['layer']>7]
     n_v4.append(d.groupby('layer_label').sum().values)

plt.plot(np.sort(it_num), np.array(n_v4))
plt.legend( list(map(bytes.decode ,d.groupby('layer_label').sum().coords['layer_label'].values)))
plt.xlabel('training iterations (~80 imgs each)' )
plt.ylabel('# TI V4 like units' )

