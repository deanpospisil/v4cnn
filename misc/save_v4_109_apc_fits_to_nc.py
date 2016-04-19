# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:42:16 2016

@author: dean
"""

import sys
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import os

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'v4cnn/common')
sys.path.append(top_dir + 'v4cnn/img_gen')
sys.path.append( top_dir + 'xarray/')

import xarray as xr

m = l.loadmat(top_dir + '/v4cnn/data/responses/V4_370PC2001.mat')
v4=m['resp'][0][0]

v4_da = xr.DataArray(v4, dims=['unit','shapes']).chunk()
#adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321]
a = np.hstack((range(14), range(18,318)))
a = np.hstack((a, range(322, 370)))
v4_da = v4_da[:, a]
v4_da = v4_da.to_dataset('resp')
v4_da.to_netcdf(top_dir + 'v4cnn/data/responses/V4_362PC2001.nc')

m = l.loadmat(top_dir + '/v4cnn/data/an_results/V4_370PC2001_LSQnonlin.mat')
v4fits = m['fI'][0][0]
#orientation, mean, curvature mean, orientation sd, curvature sd
v4_da = xr.DataArray(v4fits, dims=['unit','param'],
                     coords=[range(109), ['mori','mcurv', 'sdori', 'sdcurv', 'r' ]])
v4_da = v4_da.to_dataset('fit')
v4_da.to_netcdf(top_dir + 'v4cnn/data/an_results/V4_370PC2001_LSQnonlin.nc')



m = l.loadmat(top_dir + '/v4cnn/data/an_results/V4_370PC2001_LSQnonlin.mat')
v4fits = m['fI'][0][0]
#orientation, mean, curvature mean, orientation sd, curvature sd
v4_da = xr.DataArray(v4fits, dims=['unit','param'],
                     coords=[range(109), ['mori','mcurv', 'sdori', 'sdcurv', 'r' ]])
v4_da = v4_da.to_dataset('fit')
v4_da.to_netcdf(top_dir + 'v4cnn/data/an_results/V4_370PC2001_LSQnonlin.nc')