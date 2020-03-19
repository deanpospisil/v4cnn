# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:34:33 2018

@author: deanpospisil
"""

import scipy as sc
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
unit = 0
out = 0
fill = 1
df = sc.io.loadmat('/Users/deanpospisil/Downloads/f_o_dat.mat')['resp']

df = xr.DataArray(df,dims=['of','unit','stim','trial' ], coords=[range(dim) for dim in df.shape])
df = df.dropna('trial', how='all').transpose('unit','of','stim','trial')
df = np.sqrt(df)
m = df.mean('trial', skipna=True)
d = m.var('stim')
v = df.var('trial', skipna=True)
s = v.mean('stim')

r = [np.corrcoef(cell[0],cell[1])[0,1]**2 for cell in m]







