# -*- coding: utf-8 -*-
"""
Created on Fri May 27 01:03:44 2016

@author: deanpospisil
"""
import pandas.rpy.common as com
import sys
import os

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')

import xarray as xr
import numpy as np
fn = top_dir + 'data/models/' + 'apc_models_362.nc'
a = xr.open_dataset(fn)['resp'].values
b = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc',)['resp'].values
#xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc',)['resp'].values

#r_dataframe = com.convert_to_r_dataframe(xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc',)['resp'].to_pandas())
#r_dataframe.to_csvfile('for_r_mod.csv')

np.savetxt('for_r_mod.csv', a, delimiter=",")
np.savetxt('for_r_dap.csv', b, delimiter=",")