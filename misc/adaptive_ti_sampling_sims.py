#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:33:03 2017

@author: dean
"""
import xarray as xr
import numpy as np
import d_net_analysis as dn
import matplotlib.pyplot as plt

data_dir = '/loc6tb/'
v4_resp_ti = xr.open_dataset(data_dir + 
                             'data/responses/v4_ti_resp.nc')['resp'].load()
def interleave(x, y):
    interleaved = []
    i=0
    j=0
    for k in range(len(x)+len(y)):
        if k%2:
            interleaved.append(x[i])
            i+=1
        else:
            interleaved.append(y[j])
            j += 1
    return interleaved        

all_test = []
for n_stim in range(3, 40):
    n_stim_test = []
    for cell in v4_resp_ti:
        temp = cell.dropna('shapes', 'all').dropna('x', 'all')
        center = int(temp.var('shapes').argmax().values)
        center_resp = temp.isel(x=center)
        x = center_resp.argsort().values
        out_to_center = interleave(x, x[::-1])
        ti = dn.norm_cov(temp.values[:, out_to_center[:n_stim]].T)
        n_stim_test.append(ti)
    all_test.append(n_stim_test)
#%%
all_test = np.array(all_test)
plt.plot(np.array(range(3, 40)), all_test[-1,:] - all_test);
plt.xlabel('Number Stimuli');
plt.ylabel('Difference from TI estimate of all trials');
