# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 08:53:43 2016

@author: deanpospisil
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
import pickle as pk


cnn_names = [
'APC362_scale_1_pos_(-7, 7, 15)_ref_iter_0',
'APC362_scale_0.45_pos_(-7, 7, 15)_ref_iter_0',
'APC362_scale_0.45_pos_(-50, 48, 50)_ref_iter_0',
'APC362_scale_1_pos_(-50, 48, 50)_ref_iter_0',
]
cnn_name = cnn_names[0]
v4_name = 'V4_362PC2001'

v4ness_alex_name = top_dir + 'data/an_results/reference/v4ness_' + cnn_name + '.p'
v4ness_v4_name = top_dir + 'data/an_results/reference/v4ness_' + v4_name + '.p'

pk.load(open(v4ness_alex_name, 'rb'))
pk.load(open(v4ness_v4_name, 'rb'))

