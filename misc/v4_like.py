# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:38:45 2016

@author: dean
"""

#TI



import xray as xr
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)

# effect of blur
#dm = xr.open_dataset(cwd +'/responses/apc_models.nc', chunks={'models': 100} )
da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-120.0_120.0_10_y_-120.0_120.0_10.nc', chunks={'x': 1, 'unit':100} )
#da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_blur_0.1_2.0_10.nc', chunks={'blur': 1} )



unit_sel = np.arange(1, da.dims['unit'], 1000 )
da = da.sel(unit = unit_sel, method = 'nearest' )
da = da.sel(y = 0, method = 'nearest' )

da_n = da - da.mean('shapes')
da_n = da / np.sqrt( ( da['resp']**2 ).sum('shapes') )

fitm = (da_n*dm).sum('shapes').max('models')

