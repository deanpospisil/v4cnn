# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:14:56 2016

@author: deanpospisil
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:21:03 2016

@author: dean
"""
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
dm = xr.open_dataset(cwd +'/responses/apc_models.nc', chunks={'models': 100} )
da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc', chunks={'x': 1, 'unit':100} )

#da = da.sel(x = 0, method = 'nearest' )
#da = da.sel(y = 0, method = 'nearest' )
unit_sel = np.arange(0, da.dims['unit'],100 )
da = da.sel(unit = unit_sel, method = 'nearest' )

x_sel = np.arange(0, da.dims['x'], 3 )
da = da.sel(unit = x_sel, method = 'nearest' )

rpt = da['resp'].values
rpt= rpt.transpose(2,0,1)

u, s, v = np.linalg.svd(rpt, full_matrices=False)
print('svd')
sda = xr.DataArray( u[:,:,0], coords = [ da.coords['unit'], da.coords['shapes'] ], dims=['unit','shapes'])

sda['layer_label'] =  da.coords['layer_label']

fitm = (sda*dm).sum('shapes').max('models').load()
#columns of u will be the unit length feature sensitvity 
#rows of v will be the unit length receptive fields, across transformation
#s is how much variance each of these transformation invariant receptive fields accounts for the variance of the response

fitm.to_netcdf(cwd +'/responses/apc_models_r_svd.nc')



