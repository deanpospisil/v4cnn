# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:12:20 2017

@author: deanpospisil
"""

import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr;import pandas as pd
#import apc_model_fit as ac
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import kurtosis

#%%
da = xr.open_dataset(top_dir + '/data/responses/bvlc_reference_caffenety_test_APC362_pix_width[32.0]_x_(104.0, 124.0, 11)_x_(104.0, 124.0, 11)_amp_None.nc')['resp']
da = da.squeeze()
#%%
da = da.load()
#%%
da = da - da[0, ...]
da = da[1:, ...]
#da = da - da.mean('shapes')
#%%
ti=[]
covs = []
for unit_ind in range(96):
    unit_resp = da[:, :, :, unit_ind].values.reshape(370,11**2)
    unit_resp = unit_resp.T
    unit_resp = unit_resp - np.mean(unit_resp, 0)
    
    cov = np.cov(unit_resp, ddof=0)
    covs.append(cov)
    cov[np.diag_indices_from(cov)] = 0
    numerator = np.sum(np.triu(cov))

    vlength = np.linalg.norm(unit_resp, axis=1)
    max_cov = np.outer(vlength.T, vlength)
    max_cov[np.diag_indices_from(max_cov)] = 0
    denominator= np.sum(np.triu(max_cov))
    frac_var = numerator/denominator
    ti.append(frac_var)
ti[np.isnan(ti)] = 0


plt.imshow(cov)
plt.colorbar()
print(frac_var)
#%%
ti_x=[]
kshapes_x = []
kpos_x = []
covs_x = []

for unit_ind in range(96):
    unit_resp = da[:, :, 5, unit_ind].values
    unit_resp = unit_resp.T
    unit_resp = unit_resp - np.mean(unit_resp, 1, keepdims=True)

    cor = np.corrcoef(unit_resp)
    cov = np.dot(unit_resp, unit_resp.T)
    covs_x.append(cor)
    cov[np.diag_indices_from(cov)] = 0
    numerator = np.sum(np.triu(cov))
    
    vlength = np.linalg.norm(unit_resp, axis=1)
    max_cov = np.outer(vlength.T, vlength)
    max_cov[np.diag_indices_from(max_cov)] = 0
    denominator= np.sum(np.triu(max_cov))
    frac_var = numerator/denominator
    ti_x.append(frac_var)
    kshapes_x.append(kurtosis(np.sum(unit_resp**2, 0)))
    kpos_x.append(kurtosis(np.sum(unit_resp**2, 1)))
covs_x = np.array(covs_x)
#ti_x[np.isnan(ti_x)] = 0
plt.imshow(covs_x.mean(0), interpolation='nearest')
plt.colorbar()
#%%
ti_y = []
kshapes_y = []
kpos_y = []

for unit_ind in range(22096):
    unit_resp = da[:, 5, :, unit_ind].values
    unit_resp = unit_resp.T
    unit_resp = unit_resp - np.mean(unit_resp, 1, keepdims=True)

    
    cov = np.dot(unit_resp, unit_resp.T)
    cov[np.diag_indices_from(cov)] = 0
    numerator = np.sum(np.triu(cov))
    
    vlength = np.linalg.norm(unit_resp, axis=1)
    max_cov = np.outer(vlength.T, vlength)
    max_cov[np.diag_indices_from(max_cov)] = 0
    denominator= np.sum(np.triu(max_cov))
    
    frac_var = numerator/denominator
    
    ti_y.append(frac_var)
    kshapes_y.append(kurtosis(np.sum(unit_resp**2, 0)))
    kpos_y.append(kurtosis(np.sum(unit_resp**2, 1)))
ti_y[np.isnan(ti_y)] = 0

##%%
#ax = plt.subplot(111)
#ax.scatter(ti_x, ti_y,s=1, edgecolors='none')
#ax.set_xlabel('ti just x')
#ax.set_ylabel('ti_xy')
#ax.plot([0,1],[0,1], color='red')
#print(np.corrcoef(ti_x,ti))
#%%
non_nan = (np.isnan(ti_x)+np.isnan(ti_y))<1
non_k = (np.array(kpos_y)<6)*(np.array(kshapes_x)<40)*(np.array(kpos_x)<6)*(np.array(kshapes_y)<40)

ax = plt.subplot(111)
ax.scatter(np.array(ti_x)[non_k], np.array(ti_y)[non_k],s=1, edgecolors='none')
ax.set_xlabel('ti just x')
ax.set_ylabel('ti just y')
ax.plot([0,1],[0,1], color='red')
print(np.corrcoef(np.array(ti_x)[non_nan], np.array(ti_y)[non_nan]))
#%%
ax = plt.subplot(111)
ax.scatter(np.array(ti_x), np.array(ti_y),s=1, edgecolors='none')
ax.set_xlabel('ti just x')
ax.set_ylabel('ti just y')

#%%
plt.scatter(kshapes_x, ti_x, s=1, edgecolors='none')
#%%
non_k = np.array(kshapes_x) < 6
plt.scatter(np.array(kpos_x)[non_k], np.array(ti_x)[non_k], s=1, edgecolors='none')
#%%
sort_ind = np.argsort(ti_x)
best = sort_ind[non_k[sort_ind] * non_nan[sort_ind]][-1]
plt.plot(da[:, :, 5, list(range(22096))[best]])



