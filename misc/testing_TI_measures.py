'''
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:36:44 2016

@author: deanpospisil
"""

#comparing methods

import numpy as  np
import scipy.io as  l
import os, sys
#
import matplotlib as mpl
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm
import pandas as pd
import matplotlib.pyplot as plt

def measure_TIA(unit):
    unit = unit.dropna('x', 'all').dropna('shapes', 'all')
    tot_var = (unit**2).sum()
    s = np.linalg.svd(unit.values, compute_uv=False)[0]
    return (s**2)/tot_var

def permute_unit(unit):
    unit = unit.dropna('x', 'all').dropna('shapes', 'all')
    for x in range(len(unit.coords['x'])):
        unit[x,:] = np.random.permutation(unit[x,:].values)
    return unit

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def normalize_w_f(tia,f):
    return (tia-f)/(1-f)

#making mean 0, perfectly translation variant unit
min_ti_unit = np.concatenate([np.imag(np.fft.rfft(np.eye(5)))[:,1:], np.real(np.fft.rfft(np.eye(5)))[:,1:]], axis =1)
min_ti_unit = np.concatenate([min_ti_unit, np.zeros((51,4))])
the_nans = (np.nan*np.ones((np.shape(min_ti_unit)[0],1)))
min_ti_unit = np.concatenate([min_ti_unit, the_nans], 1)
max_ti_unit = np.tile(min_ti_unit[:,0],(5,1)).T

min_ti_unit = np.expand_dims(min_ti_unit.T, 0)
min_ti_unit = xr.DataArray(min_ti_unit, dims=['unit','x','shapes'])

max_ti_unit = np.expand_dims(max_ti_unit.T, 0)
max_ti_unit = xr.DataArray(max_ti_unit, dims=['unit','x','shapes'])


fn = top_dir +'data/responses/v4_ti_resp.nc'
v4 = xr.open_dataset(fn)['resp'].load()

v4 = v4.transpose('unit', 'x', 'shapes')
v4 = xr.concat([min_ti_unit, max_ti_unit, v4], dim='unit')

v4 = v4 - v4.mean('shapes')

#f = 0
tia = np.array([measure_TIA(unit) for unit in v4])

#f = 1/n_x
n_x = np.array([cell.dropna('x', 'all').dropna('shapes', 'all').shape[0] for cell in v4])
f = 1/n_x
tin_f_nx = (tia-f)/(1-f)


#f = max(rf)/tot_var
max_rf_var = (v4.dropna('x', 'all')**2).sum('shapes').max('x')
tot_var = (v4**2).sum(['shapes','x'])
f = (max_rf_var/tot_var).values
tin_f_maxrf = (tia-f)/(1-f)

#f = max(max(rf), max(sf))
max_rf_var = (v4.dropna('x', 'all')**2).sum('shapes').max('x')
max_srf_var = (v4.dropna('shapes', 'all')**2).sum('x').max('shapes')
max_both = xr.concat([max_rf_var,max_srf_var], dim='c').max('c')

f = max_both/tot_var
tin_f_max_rf_srf = ((tia-f)/(1-f)).values



#f = mean(TI(cell resp permuted))
n_perms = 20
tias_mean = []
tias_min = []
for unit in v4:
     unit = unit.dropna('x', 'all').dropna('shapes', 'all')
     all_perm_tia = [measure_TIA(permute_unit(unit)) for ind in range(n_perms)]
     tias_mean.append(np.mean(all_perm_tia))
     tias_min.append(np.min(all_perm_tia))

f = np.array(tias_mean)
tin_f_tias_mean = (tia-f)/(1-f)

f = np.array(tias_min)
tin_f_tias_min = (tia-f)/(1-f)

tins_lst = np.array([tia, tin_f_nx, tin_f_maxrf, tin_f_max_rf_srf, tin_f_tias_min, tin_f_tias_mean]).T
tins_nm = ['tia', 'nx', 'maxrf', 'max_rf&srf', 'tias_min', 'tias_mean']
tins = pd.DataFrame(tins_lst, columns=tins_nm)

from pandas.tools.plotting import scatter_matrix

plt.close('all')
plt.figure(figsize=(15,15))
axes = scatter_matrix(tins[2:], alpha=1, figsize=(6, 6),s=5, color='r')
for i, axes1 in enumerate(axes):
    for j, ax in enumerate(axes1):
        if i!=j:
            ax.plot([-1,1],[-1,1], alpha=0.1, color='k');
            ax.plot([-1,1],[0,0],alpha=0.1, color='k');
            ax.plot([0,0], [-1,1],alpha=0.1, color='k');

            ax.set_xlim(-0.2,1.1);ax.set_ylim(-0.2,1.1);
            ax.scatter(tins.loc[:1].values[:,j],tins.loc[0:1].values[:,i], color=['b','g'], s=5)
        else:
            ax.set_xlim(-0.2,1.1);

plt.tight_layout()
'''
import xarray as xr
def kurtosis(da):
    da = da.dropna('shapes')
    da = da.transpose('shapes','unit')
    mu = da.mean('shapes')
    sig = da.reduce(np.nanvar, dim='shapes')
    k = (((da - mu)**4).sum('shapes',skipna=True)/da.shapes.shape[0])/(sig**2)
    return k
#lets find kurtosis of our SP
fn = top_dir +'data/responses/v4_ti_resp.nc'
v4 = xr.open_dataset(fn)['resp'].load()
v4 = v4.transpose('unit', 'x', 'shapes')
v4 = v4 - v4.mean('shapes')
sp = v4.var('x')
k = kurtosis(sp)

#f = 0
tia = np.array([measure_TIA(unit) for unit in v4])
max_rf_var = (v4.dropna('x', 'all')**2).sum('shapes').max('x')
tot_var = (v4**2).sum(['shapes','x'])
f = (max_rf_var/tot_var).values
tin_f_maxrf = (tia-f)/(1-f)
print(sp.shape)
print(tin_f_maxrf.shape)
plt.scatter(k, tia);plt.ylim(0,1);plt.ylabel('tia');plt.xlabel('k');
print('r = ' +  str(np.corrcoef([k,tia])[0,1]))

plt.figure()
plt.scatter(k, tin_f_maxrf);plt.ylim(0,1);plt.ylabel('tin f=max_rf_var');plt.xlabel('k');
print('r = ' +  str(np.corrcoef([k,tin_f_maxrf])[0,1]))
