# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:33:49 2017

@author: deanpospisil
"""

import sys, os
top_dir = os.getcwd().split('v4cnn')[0] 
sys.path.append(top_dir + 'v4cnn/common/')
sys.path.append( top_dir + 'xarray/')
import xarray as xr
import apc_model_fit as ac
import d_net_analysis as dn
import numpy as np
import d_misc as dm
from scipy import io
import matplotlib.pyplot as plt

import d_net_analysis as na

data = 'v4cnn/data/'
dmod = xr.open_dataset(top_dir + data + 'models/apc_models_362_16X16.nc',
                       chunks = {'models':500, 'shapes': 370})['resp']
da = io.loadmat(top_dir + data + 'responses/cadieu_109.mat')['all']
resp = np.reshape(da, (65, 109, 368), order='F')
da = xr.DataArray(resp, dims=['x', 'unit','shapes'], coords=[range(0, 65*4, 4), range(109),  range(368)])
da = da.transpose('unit', 'x', 'shapes')
ti =na.ti_in_rf(da, stim_width=64)
#%%
plt.hist(ti, bins=1000, histtype='step', cumulative=True)
plt.xlim(0,1)

#da = da.reshape(1,65,368)
#da = xr.DataArray(da, dims=['unit', 'x', 'shapes'])
##cor = ac.cor_resp_to_model(da_c, dmod)
#rf = dn.in_rf(da, 32)
#rf[10:50] = 1
#ti = dn.ti_av_cov(da[:,12:30,:], rf=None)
#print(ti)
#plt.plot(da.sum('shapes').squeeze())