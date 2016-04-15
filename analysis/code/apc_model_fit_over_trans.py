# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:52:09 2016

@author: deanpospisil
"""
#apc model fit over trans
import sys, os
top_dir = os.getcwd().split('net_code')[0] 
sys.path.append(top_dir + 'net_code/common/')
sys.path.append( top_dir + 'xarray/')
import xarray as xr
import apc_model_fit as ac
data = 'net_code/data/'

dmod = xr.open_dataset(top_dir + data + 'models/apc_models.nc',
                       chunks={'models':1000, 'shapes':370})['resp']

da = xr.open_dataset(top_dir + data + '/responses/' 
                     'PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc', 
                     chunks={'unit':100})['resp']

#da = da.sel(x=[-5,0,5], method='nearest')
#da = da.sel(unit=range(109), method='nearest')

cor = ac.cor_resp_to_model(da, dmod, fit_over_dims=('x',))
ds = xr.Dataset({'r':cor})
ds.to_netcdf(top_dir + data + 'an_results/apc_model_fit_over_trans.nc')



