# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:10:00 2016

@author: dean
"""

import numpy as  np
import scipy.io as  l
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as mtick
#
import matplotlib as mpl
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm
import cPickle as pk
import pandas as pd

font = {'size' : 25}
mpl.rc('font', **font)

cnn_name = 'APC362_scale_1_pos_(-7, 7, 15)_ref_iter_0'
v4_name = 'V4_362PC2001'
save_folder = top_dir + 'data/an_results/reference/'

coef_var_v4 = pk.load(open(save_folder + 'coef_var' + v4_name, 'rb'))
coef_var_alex = pk.load(open(save_folder +'coef_var' + cnn_name, 'rb'))
eye_r2_v4 = pk.load(open(save_folder  + 'eye_r2_' + v4_name, 'rb'))
eye_r2_alex = pk.load(open(save_folder  + 'eye_r2_' + cnn_name, 'rb'))
k_alex = pk.load(open(save_folder  + 'k_' + cnn_name, 'rb'))
k_v4 = pk.load(open(save_folder  + 'k_' + v4_name, 'rb'))
ti_v4 = pk.load(open(save_folder  + 'ti_'+ v4_name, 'rb'))
ti_alex = pk.load(open(save_folder  + 'ti_'+ cnn_name, 'rb'))
tilc_alex = pk.load(open(save_folder  + 'trans_ill_cond_' + cnn_name, 'rb'))

v4 = pd.concat( [coef_var_v4, eye_r2_v4, k_v4], axis=1)
cnn = pd.concat([coef_var_alex, eye_r2_alex, k_alex, ti_alex, tilc_alex], axis=1)

import re
[]
lay_type_index = [re.findall('[^0-9]+', label)[0]
                 for label in cnn.index.get_level_values('layer_label')]
rect_or_not = ['relu' in label
                 for label in cnn.index.get_level_values('layer_label')]
cnn['lay_type'] = lay_type_index
cnn.set_index('lay_type', append=True, inplace=True)
cnn['relu'] = rect_or_not
cnn.set_index('relu', append=True, inplace=True)
cnn.reorder_levels(['lay_type', 'layer_label', 'relu', 'layer_unit'])



