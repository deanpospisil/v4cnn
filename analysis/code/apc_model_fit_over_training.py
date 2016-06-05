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
import d_misc as dm

data = 'v4cnn/data/'


dmod = xr.open_dataset(top_dir + data + 'models/apc_models_362_16X16.nc',
                       chunks = {'models': 500, 'shapes': 370}  )['resp']
#dmod = dmod.sel(models = range(10), method = 'nearest' )
#ds = xr.open_mfdataset(top_dir + data + 'responses/' +'iter_*.nc', 
#                       concat_dim = 'niter', chunks = {'unit':100, 'shapes': 370})
#ds = xr.open_mfdataset('/data/dean_data/responses/' +'iter_*.nc', 
 #                      concat_dim = 'niter', chunks = {'unit':100, 'shapes': 370})
#ds = ds.sel(x = np.linspace(-50, 50, 2), method = 'nearest' )
#ds = ds.sel(niter=np.linspace(0, da.coords['niter'].shape[0], 2),  
#                                method = 'nearest')
#ds = ds.sel(unit=range(10), method='nearest')
                       
substrings = [ 'iter_small_trans*.nc']  
for substring in substrings:  
    all_iter = dm.list_files('/data/dean_data/responses/' + substring)
    
    for i, itername in enumerate(all_iter):
        print(itername)
        #    da_c = ds.sel(niter=iterind)['resp']
        da_c = xr.open_dataset(itername)['resp']
        da_c = da_c.sel(x=0, method='nearest').squeeze().chunk({'unit':50,'shapes': 370})
        #    cor = ac.cor_resp_to_model(da_c, dmod, fit_over_dims=('x',))
        cor = ac.cor_resp_to_model(da_c, dmod)
        cor.to_dataset(name='r').to_netcdf(top_dir + 'v4cnn/data/an_results/noTI_r_' 
        + itername.split('responses/')[1])
    
    #ds = xr.open_mfdataset(top_dir + 'analysis/data/an_results/r_iter_*.nc', concat_dim = 'niter')
    #ds.to_netcdf(top_dir + 'analysis/data/r_iter_total_' + str(da.niter.shape[0]) + '.nc')
