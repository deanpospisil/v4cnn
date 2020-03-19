# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:54:45 2018

@author: deanpospisil
"""

import numpy as np
import matplotlib.pyplot as plt
from  scipy.stats import beta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 20
df1 = 1
df2 = n-1
a = df1
b = df2

x = np.linspace(0, 1, 100)

plt.plot(x, beta.pdf(x, df1/2, df2/2),
         'r-', lw=5, alpha=0.6, label='beta pdf')


import pandas as pd
def norm_av_cov(unit, return_num_den=False):
    unit = unit.transpose('shapes','x')
    unit = unit.dropna('x', 'all').dropna('shapes', 'all').values
    cov = np.dot(unit.T, unit)
    cov[np.diag_indices_from(cov)] = 0
    numerator = np.sum(np.triu(cov))
    vlength = np.linalg.norm(unit, axis=0)
    max_cov = np.outer(vlength.T, vlength)
    max_cov[np.diag_indices_from(max_cov)] = 0
    denominator= np.sum(np.triu(max_cov))
    if return_num_den:
        return numerator, denominator
    else:
        return numerator/denominator
    
top_dir = '/Users/deanpospisil/Desktop/modules/v4cnn/'
import xarray as xr
import numpy as np
#import matplotlib.pyplot as plt 
import scipy.stats as stats

fn = top_dir +'data/responses/v4_ti_resp.nc'
v4 = xr.open_dataset(fn)['resp'].load()
log_v4 = np.log(v4+0.001)
t = [norm_av_cov(v4[i] - v4[i].mean('shapes')) for i in range(80)]



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
unit = 20
unit = v4.isel(unit=unit).dropna('x', 'all').dropna('shapes', 'all')
p0 = unit.isel(x=0)
p1 = unit.isel(x=1)
p2 = unit.isel(x=2)

ax.scatter(p0,p1,p2)
