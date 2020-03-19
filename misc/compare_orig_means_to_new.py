# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 09:55:12 2017

@author: deanpospisil
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import xarray as xr 

v4_resp_mean = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc')['resp'].load()
v4_resp_trials = xr.open_dataset(top_dir + 'data/responses/apc_orig/apc370_with_trials.nc')
v4_resp_trials = v4_resp_trials['resp'].load().transpose('shapes','unit','trials')
v4_resp_trials = v4_resp_trials.loc[v4_resp_mean.coords['shapes'].values, :,:]

#lets compare the first two cells
for cell in range(109):
    cell_1 = v4_resp_mean[:,cell]
    cell_2 = v4_resp_trials[:,cell,:].var('trials')
    print(np.corrcoef(cell_1, cell_2)[0,1])
    

#%%
#lets compare the first two cells
for cell in range(109):
    cell_mean = v4_resp_trials[:,cell,:].mean('trials')
    cell_var = v4_resp_trials[:,cell,:].mean('trials')

ax = plt.subplot(111)
plt.scatter(v4_resp_trials.var('trials',skipna=True).values.ravel(), 
            v4_resp_trials.mean('trials',skipna=True).values.ravel(), s=1, lw=0)
ax.set_xlabel('var');ax.set_ylabel('mean');
plt.gca().set_aspect('equal', 'box-forced')
ax.plot([0,50],[0,50]);ax.set_xlim(-10,320);ax.set_ylim(0,40);
ax.set_yticks([0,20,40])




    