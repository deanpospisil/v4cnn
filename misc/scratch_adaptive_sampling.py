#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 10:06:57 2018

@author: dean
"""


import numpy as np
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir + 'v4cnn'
import xarray as xr
import pandas as pd
import pickle



import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, RidgeCV, LassoLarsIC, Ridge
from sklearn import datasets
from sklearn.utils import resample
def dyn_range_calc(x):
    #assumes we have the means of the response in a single vector
    sx = np.sqrt(x)
    dsx = sx - np.mean(sx)
    return np.mean(dsx**2)

top_dir = '/'
v4_resp_trials = xr.open_dataset('/home/dean/Desktop/v4cnn/data/responses/apc_orig/apc370_with_trials.nc')['resp']
v4 = v4_resp_trials[:, 1:]

net_resp_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370.nc'
da = xr.open_dataset('/loc6tb' + '/data/responses/v4cnn/' + net_resp_name)['resp']

da_c = da.sel(x=114).squeeze()
X = da_c[1:, da_c.coords['layer_label'].values=='conv5'].values
X = X - X.mean(0)
#%%
#take the svd of our cnn's resps to the stimuli
y = v4[55]

u, s, v = np.linalg.svd(X, full_matrices=0)
#get m rand
n_beg_samps = 10
trial_len = n_beg_samps
n_rep_samps = np.round(trial_len/3.)



n_tot_stim = X.shape[0]
 
first_stims = np.random.choice(n_tot_stim, n_beg_samps, replace=False)
cur_stims = first_stims

cur_y = y[cur_stims, 0]
cur_u = u[cur_stims]

#as a rule of thumb we will keep p<n/2


#%%
n_samps = 10
mod = Ridge(alpha=.1, fit_intercept=False, normalize=True)
Xs = u[:n_samps]
ys = y[:n_samps,0].values
Xs = np.eye(len(ys))
#Xs = Xs - Xs.mean(0, keepdims=True)
ys = ys - ys.mean(0, keepdims=True)


mod.fit(Xs, ys)
mod.score(Xs, ys)

#v.shape
#%%
samples_per_step = 5
dyn_range_all = np.array([dyn_range_calc(cell.values) for cell in v4_resp_trials.mean('trials')])
units = dyn_range_all.argsort()[::-1][:3]
units_dyn_range = dyn_range_all[units]


beg_samps = 20
n_sample_steps = 50
n_ad_sample_runs = 10
n_units = len(units)

sim_dat = np.zeros((n_units, n_ad_sample_runs, n_sample_steps))
sim_dat_ran = np.zeros((n_units, n_ad_sample_runs, n_sample_steps))
sim_dat_alpha = np.zeros((n_units, n_ad_sample_runs, n_sample_steps))
for i, unit in enumerate(units):
    print('unit: ' + str(i))
    y = v4_resp_trials.isel(unit=unit).mean('trials').values
    ind = range(len(y))
    for j in range(n_ad_sample_runs):
        print('run: '+ str(j))
        start_ind_r, start_x_r, start_y_r = resample(ind, X, y, n_samples=beg_samps, replace=False)
        start_x_m = start_x_r[:]
        start_y_m = start_y_r[:]
        start_ind_m = start_ind_r[:]
        for k in range(n_sample_steps):
            #model sampling
            dyn_range_m = dyn_range_calc(start_y_m)
            sim_dat[i,j,k] = dyn_range_m
            
            model = LassoCV(cv=3, n_jobs=-1, n_alphas=30, normalize=True).fit(
                    start_x_m, start_y_m)
            #model = LassoLarsIC(normalize=True).fit(
            #        start_x_m, start_y_m);
            sim_dat_alpha[i,j, k] = model.alpha_
            unsampled_inds = list(set(range(len(y)))-set(start_ind_m))
            pred = model.predict(X[unsampled_inds])
            pred_sq_dif = np.abs(np.sqrt(pred) - np.mean(np.sqrt(start_y_m)))
            #pred_sq_dif = np.abs(np.sqrt(y[unsampled_inds]) - np.mean(np.sqrt(start_y_m)))
            best_dif_inds = list(np.argsort(pred_sq_dif)[::-1][:samples_per_step])
            next_y_ind = list(np.array(unsampled_inds)[best_dif_inds])
            start_ind_m = start_ind_m + next_y_ind
            start_x_m = X[start_ind_m]
            start_y_m = y[start_ind_m]            
            #random sampling
            dyn_range_r = dyn_range_calc(start_y_r)
            sim_dat_ran[i,j,k] = dyn_range_r
            dyn_range_r = dyn_range_calc(start_y_r)
            unsampled_inds = list(set(range(len(y)))-set(start_ind_r))
            next_y_ind = resample(unsampled_inds, n_samples=samples_per_step, replace=False)[:]
            start_ind_r = start_ind_r + next_y_ind
            start_y_r = y[start_ind_r]
