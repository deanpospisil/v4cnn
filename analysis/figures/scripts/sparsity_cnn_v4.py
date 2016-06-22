# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:32:56 2016

@author: dean
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')

import xarray as xr
import d_misc as dm
import re
def da_coef_var(da):
    da_min_resps = da.min('shapes')
    if ((da<0).values).any():
        da[:,da_min_resps<0] = da[:,da_min_resps<0] - da_min_resps[da_min_resps<0]
    mu = da.mean('shapes')
    sig = da.reduce(np.std, dim='shapes')
    return 1./(((mu/sig)**2)+1)

fn = top_dir + 'data/models/' + 'apc_models_362.nc'
#dmod = xr.open_dataset(fn, chunks={'models': 100, 'shapes': 370}  )['resp']
#daa = xr.open_dataset(top_dir + 'data/an_res/APC362_scale_1_pos_(-7, 7, 15)_iter_450000.nc')['resp']
#daa = daa.sel(x=0)
#
#dam = daa.min('shapes')
#daa[:,dam<0] = daa[:,dam<0] - dam[dam<0]

#inds = degen(daa).values
#indsv = degen(da).values
#def softmax(w):
#
#    return np.exp(w)/np.sum(np.exp(w))
#make all positive
#fc8 = daa.coords['layer_label']=='fc8'
#daa[:,fc8] = daa[:,fc8] - daa[:,fc8].min()
##daa[:,fc8] = abs(daa[:,fc8])
#prob =  daa.coords['layer_label']=='prob'
#daa = daa[:,-prob]
#
#alex = da_coef_var(daa)
all_iter = dm.list_files(top_dir + 'data/an_results/sparsity_APC*.nc')
iter_numbers = [int(re.findall('\d+', line)[-1]) for line in all_iter]
all_iter = [all_iter[sort_i] for sort_i in np.argsort(iter_numbers)]
spar_iter = [xr.open_dataset(name) for name in all_iter]
spar_all = xr.concat(spar_iter, xr.DataArray(np.sort(iter_numbers), dims='iter', name='iter'))
spar_all.attrs['fn'] = all_iter[0]



da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
v4 = da_coef_var(da)
alex = spar_all.isel(iter=-1)['spar']
#alex = xr.open_dataset(top_dir + 'data/an_results/sparsity_APC362_scale_1_pos_(-50, 48, 50)_iter_450000.nc')['spar']


plt.close('all')
plt.figure()
plt.subplot(212)
plt.title('Normalized Histogram Sparsity Values')
plt.hist(alex.values, range=(0,1), normed=True, color ='blue', alpha=0.5, bins=100)
plt.hist(v4.values, range=(0,1), normed=True, color ='red', alpha=0.5, bins=20)
plt.xlabel('Coefficient of Variation')
plt.legend(['Alex', 'V4'])

plt.subplot(211)
plt.title('Example V4 Response Histogram')
plt.hist(da[int(v4.argmin().values)], alpha=0.5, bins=20, range=(0,1), log=True)
plt.hist(da[int(v4.argmax().values)], alpha=0.5, bins=20, range=(0,1), log=True)
plt.xlabel('Normalized Firing Rate')
plt.legend([round(v4.min().values,2), round(v4.max().values,2)], title='Sparsity')

#plt.title('Coef Var : ' + str(np.sum( (k<np.nanmax(kv))*(k>np.nanmin(kv)) ))
#+' / '+ str(len(k))+ ' units left, and ' + str(np.sum(overlap*-inds)) + ' / ' +str(np.sum(overlap)) +' with 50% frac var req')
plt.tight_layout()

plt.savefig(top_dir + 'analysis/figures/images/sparsity_measure_plot.eps')

v4 = da_coef_var(da)
v4.to_dataset('spar').to_netcdf(top_dir + 'data/an_results/spar_v4.nc')
alex.to_dataset('spar').to_netcdf(top_dir + 'data/an_results/spar_alex_last.nc')

