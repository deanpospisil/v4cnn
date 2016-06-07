# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 12:50:36 2016

@author: dean
"""

import sys, os
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'v4cnn/common/')
sys.path.append( top_dir + 'xarray/')
import xarray as xr
import numpy as np
import d_misc as dm

data = 'v4cnn/data/'

all_iter = dm.list_files('/data/dean_data/responses/' +'iter_*.nc')

for i, itername in enumerate(all_iter):
    print(itername)
#    da_c = ds.sel(niter=iterind)['resp']
    da_c = xr.open_dataset(itername, chunks = {'unit':100,'shapes': 370}  )['resp']
    da_c = da_c - da_c.mean(['shapes'])
    s = np.linalg.svd(da_c.values.T, compute_uv=0)
    best_r_alex = np.array([(asingval[0]**2)/(sum(asingval**2)) for asingval in s])
    ti = xr.DataArray(best_r_alex).reindex_like(da_c.sel(x=0, method='nearest'))

    ti.to_dataset(name='ti').to_netcdf(top_dir + 'v4cnn/data/an_results/translation_invariance_'
    + itername.split('responses/')[1])

#ds = xr.open_mfdataset(top_dir + 'analysis/data/an_results/r_iter_*.nc', concat_dim = 'niter')
#ds.to_netcdf(top_dir + 'analysis/data/r_iter_total_' + str(da.niter.shape[0]) + '.nc')
