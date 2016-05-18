# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:19:38 2016

@author: deanpospisil
"""

import os, sys
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir + 'xarray')
import xarray as xr
fname = 'apc_models_r_trans101'
fitm = xr.open_dataset(top_dir +'data/an_results/' + fname + '.nc' )

