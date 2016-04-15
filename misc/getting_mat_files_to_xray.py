# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:10:19 2016

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
import scipy.io as l
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)



# effect of blur
dm = xr.open_dataset(cwd +'/responses/apc_models.nc', chunks={'models': 100} )
models = dm['resp'].values

mat2 = l.loadmat(cwd + '/responses/AlexNet_51rfs370PC2001.mat')
da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_blur_0.1_2.0_10.nc', chunks={'blur': 1, 'unit': 100} )

#the recent images blur of 2, normalizing using numpy methods
da= da.sel( blur = 2)
fcl = [ i for i, name in enumerate(da.coords['layer_label'].values) if str(name) == "b'fc8'" ]
da=da.sel(unit = fcl)
fcn = da['resp'].values.T
fcn = fcn - np.mean(fcn, axis= 1).reshape((1000,1))
fcn = fcn / np.linalg.norm(fcn, axis= 1).reshape((1000,1))

fitsn  = np.dot( fcn, models ).max(axis=1)

#normalizing by xray methods.
da_n = da['resp'] - da['resp'].mean('shapes')
da_n = da_n / (( da_n**2 ).sum('shapes') )**0.5
wrong = da_n.values.T

fitsx = np.dot(wrong, models).max(axis=1)


#looking the best performers from original data
fc=mat2['resp'][0][7]
fc = fc - np.mean(fc, axis= 1).reshape((1000,1))
fc = fc / np.linalg.norm(fc, axis= 1).reshape((1000,1))

fits = np.dot(fc,models).max(axis=1)






print( np.sum(fitsn>0.5) )
print( np.sum(fits>0.5)  )
print( np.sum(fitsx>0.5))
