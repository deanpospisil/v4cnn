#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:35 2018

@author: dean
"""
# Author: Olivier Grisel, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause



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
goforit=True      
#if 'netwts' not in locals() or goforit:
#    with open(top_dir + '/nets/netwts.p', 'rb') as f:    
#        try:
#            netwts = pickle.load(f, encoding='latin1')
#        except:
#            netwts = pickle.load(f)
## reshape fc layer to be spatial
#netwts[5][1] = netwts[5][1].reshape((4096, 256, 6, 6))
#wts_by_layer = [layer[1] for layer in netwts]
v4_resp_trials = xr.open_dataset(top_dir +
                                 '/data/responses/apc_orig/apc370_with_trials.nc')['resp']
data_dir = '/loc6tb'
net_resp_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370.nc'
da = xr.open_dataset(data_dir + '/data/responses/v4cnn/' + net_resp_name)['resp']
#%%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, RidgeCV, LassoLarsIC
from sklearn import datasets
from sklearn.utils import resample
def dyn_range_calc(x):
    #assumes we have the means of the response in a single vector
    sx = np.sqrt(x)
    dsx = sx - np.mean(sx)
    return np.mean(dsx**2)

samples_per_step = 5
dyn_range_all = np.array([dyn_range_calc(cell.values) for cell in v4_resp_trials.mean('trials')])
units = dyn_range_all.argsort()[::-1][:3]
units_dyn_range = dyn_range_all[units]
da_c = da.sel(x=114).squeeze()
X = da_c[:, da_c.coords['layer_label'].values=='conv5'].values
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



#%%
for unit, d_range, n_unit in zip(range(len(units)), units_dyn_range[:10], units):
    plt.figure(figsize=(3,2))
    n_samps = np.arange(n_sample_steps)*samples_per_step + beg_samps
    plt.plot(n_samps, np.mean(sim_dat_ran[unit, ...].T,1), color='g',alpha=1)
    plt.plot(n_samps, np.mean(sim_dat[unit, ...].T,1), color='r', alpha=1)
    plt.legend(['Random', 'Adaptive'])
    plt.plot(n_samps, sim_dat_ran[unit, ...].T, color='g',alpha=0.1)
    plt.plot(n_samps, sim_dat[unit, ...].T, color='r', alpha=0.1)
    #plt.plot([0, n_samps[-1]], [d_range,d_range])

    plt.xlabel('Number stimuli shown');plt.ylabel('Dynamic range index');

    unit_id = v4_resp_trials[n_unit].coords['w_lab'].values
    plt.savefig(top_dir + '/analysis/figures/images/ad_samp/'
            +str(unit_id)+'_ad_samp_cnn_v4.pdf', bbox_inches='tight')

#%%
plt.figure(figsize=(3.5, 2.8))
top_n = 30
sim_dat_mean = np.mean(sim_dat, 1)[:top_n]
ratio = sim_dat_mean/units_dyn_range.reshape(109,1)[:top_n]
plt.plot(n_samps, ratio.T, color='k', alpha=0.1)
mr = np.mean(np.max(ratio, 1))
mrt = np.round(np.mean(np.argmax(ratio,1)))
plt.plot(n_samps, np.mean(ratio,0), color='r')
plt.yticks([.75, 1, 1.25, np.round(mr,2), 1.5, 1.75 ,2]);
plt.plot([0,n_samps[int(mrt)]], [mr, mr], color='k')
plt.plot([n_samps[int(mrt)], n_samps[int(mrt)]], [.5, mr], color='k')
plt.xlim(0,250);plt.ylim(0.75,2)
plt.plot([0,250], [1,1], color='k', ls='--')
plt.annotate('Avg. # Stim.\n to Max', [n_samps[int(mrt)]+2, .77] )
plt.annotate('Average Max Ratio', [4, mr+0.03])
plt.annotate('Average Ratio', [150, 1.4], color='r')
plt.xlabel('Number stimuli shown');plt.ylabel('Dynamic range ratio\nadaptive/random');    

plt.savefig(top_dir + '/analysis/figures/images/ad_samp/ad_samp_vs_ran.pdf', bbox_inches='tight')
#%%
plt.figure(figsize=(3.5,2.5))
plt.hist(np.max(sim_dat_mean,1), normed=True, histtype='step', 
         cumulative=True, range=[.4, 3.4], bins=100, color='r')
plt.hist(units_dyn_range[:top_n], normed=True, histtype='step', 
         cumulative=True, range=[.4,3.4], bins=100, color='g')
plt.legend(['Adaptive sampling', 'Random sampling'], loc='center right')
plt.annotate('Tuning threshold', [1.05,.2])
plt.plot([1,1], [0,1], color='k');plt.yticks([0,1/4., 1/2., 3/4.,1])
plt.xlabel('Dynamic range index');plt.ylabel('Fraction units')
plt.savefig(top_dir + '/analysis/figures/images/ad_samp/cum_ad_samp_vs_ran.pdf', bbox_inches='tight')

#%%
## Display results
#m_log_alphas = -np.log10(model.alphas_)
#
#plt.figure()
#plt.plot(m_log_alphas, model.mse_path_, ':')
#plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
#         label='Average across the folds', linewidth=2)
#plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
#            label='alpha: CV estimate')
#
#plt.legend()
#
#plt.xlabel('-log(alpha)')
#plt.ylabel('Mean square error')
#plt.title('Mean square error on each fold: coordinate descent '
#          '(train time: %.2fs)' % t_lasso_cv)
#plt.axis('tight')
##plt.ylim(ymin, ymax)

