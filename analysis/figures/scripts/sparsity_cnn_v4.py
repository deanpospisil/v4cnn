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

def da_coef_var(da):
    da_min_resps = da.min('shapes')
    da[:,da_min_resps<0] = da[:,da_min_resps<0] - da_min_resps[da_min_resps<0]
    mu = da.mean('shapes')
    sig = da.reduce(np.std, dim='shapes')
    return 1./(((mu/sig)**2)+1)

fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models': 100, 'shapes': 370}  )['resp']
da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
daa = daa.sel(x=0)

dam = daa.min('shapes')
daa[:,dam<0] = daa[:,dam<0] - dam[dam<0]

#inds = degen(daa).values
#indsv = degen(da).values
#def softmax(w):
#
#    return np.exp(w)/np.sum(np.exp(w))

print('trolls')


#make all positive
#fc8 = daa.coords['layer_label']=='fc8'
#daa[:,fc8] = daa[:,fc8] - daa[:,fc8].min()
##daa[:,fc8] = abs(daa[:,fc8])

prob =  daa.coords['layer_label']=='prob'
daa = daa[:,-prob]

v4 = da_coef_var(da)
alex = da_coef_var(daa)

plt.close('all')
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
alex = da_coef_var(daa)
v4.to_dataset('spar').to_netcdf(top_dir + 'data/an_results/spar_v4.nc')
alex.to_dataset('spar').to_netcdf(top_dir + 'data/an_results/spar_alex_last.nc')


'''
#kurtosis: doesnt work with out negatives
k = st.kurtosis(da.values)
plt.scatter(2*np.ones(np.shape(k)), k)
print(np.min(k))
print(np.max(k))

k = st.kurtosis(daa.T)
plt.scatter(np.ones(np.shape(k)), k)
print(np.min(k))
print(np.max(k))

#gini
s=1
plt.close('all')
plt.subplot(211)
print('gini')
kv = map(gini, da.values)
k = np.array(map(gini, daa.T.values))
plt.hist(k[np.logical_not(np.isnan(k))], range=(-100,100), normed=True, color ='blue', edgecolor='blue', alpha=0.5, bins=1000, log=True)
plt.hist(kv, normed=True, color ='red',edgecolor='red', alpha=0.5, bins=20, log=True)
plt.legend(['Alex', 'V4'])

overlap = ((k<np.nanmax(kv))*(k>np.nanmin(kv)))


plt.title('Gini Coef : ' + str(np.sum( (k<np.nanmax(kv))*(k>np.nanmin(kv)) ))
+' / '+ str(len(k))+ ' units left, and ' + str(np.sum(overlap*-inds)) + ' / ' +str(np.sum(overlap)) + ' with 50% frac var req')

print('Number AN units within range of V4 sparsity by Gini')
print(np.sum( (k<np.nanmax(kv))*(k>np.nanmin(kv)) ))



def degen(daa):
    minfracvar = 0.5
    _ = (daa**2)
    tot_var = _.sum('shapes')
    non_zero = tot_var<1e-16
    just_one_shape = (_.max('shapes')/tot_var)>minfracvar
    degen_inds = just_one_shape + non_zero
    return degen_inds

def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area

def coef_var(x):
    mu = np.nanmean(x)
    sig = np.nanstd(x)

    return 1./((mu/sig)**2+1)
'''