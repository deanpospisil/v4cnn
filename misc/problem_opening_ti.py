# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 20:31:00 2016

@author: dean
"""

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
xr.open_dataset(top_dir+'data/an_results/ti_APC362_scale_0.45_pos_(-7, 7, 15)_iter_450000.nc')