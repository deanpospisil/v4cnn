# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:49:44 2016

@author: dean
"""
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')

import xarray as xr
import apc_model_fit as ac
import pandas as pd
import matplotlib.pyplot as plt

if 'alex_resp' not in locals():
    cnn_name = 'APC362_scale_0.45_pos_(-50, 48, 50)_ref_iter_0'
    alex_resp = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp'].load().squeeze()

alex_var = ((alex_resp)**2).sum('shapes')
in_rf = (alex_var - alex_var.to_pandas().mode(0)[:,0] ) > 0


#plt.plot(in_rf[:, :96])
plt.imshow(alex_var[:, :96], interpolation ='nearest')

plt.tight_layout()