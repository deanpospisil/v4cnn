#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:14:39 2017

@author: dean
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn')
sys.path.insert(0, top_dir + 'xarray/');
top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr
import pandas as pd
#%%

def adjust_to_r(static_vector, shift_vector, desired_r):
    #center both vectors
    if np.corrcoef(shift_vector, static_vector)[0,1]<0:
        shift_vector = -shift_vector
    b = shift_vector
    b = b - b.mean()
    x = static_vector
    x = x - x.mean()
    
    #project x onto b
    s = np.dot(x, b)/np.linalg.norm(b)**2
    b = b * s
    #get the error vector, or vecto pointing from tip of b to x
    e = x - b
    #convert desired r to angle    
    theta = np.arccos(desired_r)
    #get current angle between x and b
    phi = np.arccos(np.corrcoef(x, b)[0,1])
    #how much angle to shift b
    beta = phi - theta
    mag_b = np.linalg.norm(b)#adjacent length
    mag_e = np.linalg.norm(e)#opposite length
    #solve for git stscaling of a to the opposite side for beta
    a = (mag_b * np.tan(beta))/mag_e
    shifted_vector = a * e + b # now shift b with a*e
    return shifted_vector

v4_resp_mean = xr.open_dataset(top_dir + 
                               'data/responses/V4_362PC2001.nc')['resp'].load()
v4_resp_trials = xr.open_dataset(top_dir +
                                 'data/responses/apc_orig/apc370_with_trials.nc')
v4_resp_trials = v4_resp_trials['resp'].load().transpose('shapes','unit','trials')
v4_resp_trials = v4_resp_trials.loc[v4_resp_mean.coords['shapes'].values, :, :]
v4_resp_mean = v4_resp_trials.mean('trials')
#%%
m_cor_list = []
for cell in range(v4_resp_mean.shape[1]):
    x = v4_resp_mean[:,cell]
    nstim = x.shape[0]
    correlations = [1,]
    n_model_resample = 20
    n_noise_resample = 20
    n_trials = 5
    r_list = []

    for desired_r in correlations:
        gen_vec = np.random.randn(n_model_resample, nstim)
        for vec in gen_vec:
            shift_vec = adjust_to_r(x, vec, desired_r)
            sim_resp = np.random.poisson(x, size=(n_noise_resample, n_trials, nstim))
            r = [np.corrcoef(resp, shift_vec)[0, 1] for resp in sim_resp.mean(1)]
        r_list.append(r)

    r_list = np.abs(np.array(r_list))
    m_cor_list.append(np.mean(r_list))
#%%
_=plt.hist(np.array(m_cor_list), histtype='step')
plt.title('Distribution of correlation to perfect model after noise')
plt.xlabel('Correlation');plt.ylabel('Count');

#%%

import apc_model_fit as ac


v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir +
                              'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
v4_resp_apc = v4_resp_mean
file = open(top_dir + 'data/responses/v4_apc_109_neural_labels.txt', 'r')
wyeth_labels = [label.split(' ')[-1] for label in 
            file.read().split('\n') if len(label)>0]
v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()

apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                  dmod.chunk({}), 
                                  fit_over_dims=None, 
                                  prov_commit=False)
#%%
fit_mod = []
from scipy.stats import linregress
for i, fit in enumerate(apc_fit_v4):
    mod = dmod[:, int(fit.models.values)]
    dat = v4_resp_apc[:, i].values
    slope, inter, r, p, sterr = linregress(mod.values, dat)
    if r>0.6:
        fit_mod.append(slope*mod + inter)
#    plt.figure()
#    plt.scatter(fit_mod, dat)
#    plt.title(np.round(r,2))
#    plt.axis('equal')
    

#%%
r_thresh = [0.3, 0.5, 0.6, 0.7]
m_cor_r_thresh_list = []
for an_r_thresh in r_thresh:
    print(an_r_thresh)
    fit_mod = []
    for i, fit in enumerate(apc_fit_v4):
        mod = dmod[:, int(fit.models.values)]
        dat = v4_resp_apc[:, i].values
        slope, inter, r, p, sterr = linregress(mod.values, dat)
        if r>an_r_thresh:
            fit_mod.append(slope*mod + inter)
        
    m_cor_list = []
    for i, a_fit_mod in enumerate(fit_mod):
        #print(i)
        x = a_fit_mod
        if sum(x<0)>0:
            x = x - x.min()
        nstim = x.shape[0]
        n_noise_resample = 10
        n_trials = 5
        r_list = []
    
        shift_vec = x
        sim_resp = np.random.poisson(x, size=(n_noise_resample, n_trials, nstim))
        r = [np.corrcoef(resp, shift_vec)[0, 1] for resp in sim_resp.mean(1)]
        r_list.append(r)
    
        r_list = np.abs(np.array(r_list))
        m_cor_list.append(np.mean(r_list))
    m_cor_r_thresh_list.append(np.array(m_cor_list))

#%%
plt.figure(figsize=(4,8))
for i, m_cor_list in enumerate(m_cor_r_thresh_list):
    
    plt.subplot(len(r_thresh), 1, i+1)
    if i ==0:
        plt.title('Noise added apc model fit to V4 unit' +'\nV4 apc model fit >=' + str(np.round(r_thresh[i],2)))
    else:
        plt.title(str(np.round(r_thresh[i],2)))

    _=plt.hist(np.array(m_cor_list), histtype='step')
    plt.xlabel('Correlation');plt.ylabel('Count');
    plt.xlim(0,1)
plt.tight_layout()
#%%



#%%
os.listdir(top_dir+'/analysis/figures/images/v4cnn_cur/')










