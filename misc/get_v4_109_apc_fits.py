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

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'net_code/common')
sys.path.append(top_dir + 'net_code/img_gen')
sys.path.append( top_dir + 'xarray/')

#import d_curve as dc
#import d_misc as dm
#import base_shape_gen as bg
import pickle
import xarray as xr


import apc_model_fit as ac
maxAngSD = np.deg2rad(171)
minAngSD = np.deg2rad(23)
maxCurSD = 0.98
minCurSD = 0.09
nMeans = 16
nSD = 16

with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
    shape_dict_list2 = pickle.load(f)
fn = 'apc_for_reza_v4.nc'

dmod_old = ac.make_apc_models(shape_dict_list2, range(370), fn, nMeans, nSD,
                      maxAngSD, minAngSD, maxCurSD, minCurSD,
                      model_params_dict=None, prov_commit=False, cart=True,
                      save=False)
m = l.loadmat(top_dir + 'net_code/data/responses/V4_370PC2001.mat')

v4=m['resp'][0][0]

v4_da = xr.DataArray(v4, dims=['unit','shapes']).chunk()
cor_old = ac.cor_resp_to_model(v4_da, dmod_old.chunk(), fit_over_dims=None, prov_commit=False)