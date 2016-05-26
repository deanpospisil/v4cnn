# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:24:20 2016

@author: deanpospisil
"""
import sys, os
top_dir = os.getcwd().split('v4cnn')[0] 
sys.path.append(top_dir + 'v4cnn/common/')
sys.path.append( top_dir + 'xarray/')
import xarray as xr
import apc_model_fit as ac
import numpy as np

data = 'v4cnn/data/'


dmod = xr.open_dataset(top_dir + data + 'models/apc_models_362_16X16.nc',
                       chunks = {'models': 1000, 'shapes': 370}  )['resp']
dmod = dmod.sel(models = range(10), method = 'nearest' )
ds = xr.open_mfdataset('/home/dean/caffe/models/bvlc_reference_caffenet/caffenet_train/' +'_iter_*.nc', 
                       concat_dim = 'niter', chunks = {'unit':100, 'shapes': 370})
ds = ds.sel(x = np.linspace(-50, 50, 2), method = 'nearest' )
ds = ds.sel(niter=np.linspace(0, da.coords['niter'].shape[0], 2),  
                                method = 'nearest')
ds = ds.sel(unit=range(10), method='nearest')

for iterind in ds.niter.values:
    da_c = ds.sel(niter=iterind)
    cor = ac.cor_resp_to_model(da_c, dmod, fit_over_dims=('x',))
    cor.to_dataset(name='r').to_netcdf(top_dir + 'analysis/data/r_iter_' + str(iterind) + '.nc')

ds = xr.open_mfdataset(top_dir + 'analysis/data/an_results/r_iter_*.nc', concat_dim = 'niter')
ds.to_netcdf(top_dir + 'analysis/data/r_iter_total_' + str(da.niter.shape[0]) + '.nc')
