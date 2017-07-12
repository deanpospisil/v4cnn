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
import xarray as xr
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
#divide by the duration in seconds to get spikes per second.
v4_resp_mean = v4_resp_trials.mean('trials')/(v4_resp_trials.attrs['dur']/1000.)
#%%
n_noise_resample = 100 #how many times to repeat the simulation
n_trials = 5
apc_m_cor_list = []
apc_sd_cor_list = []
for cell in range(v4_resp_mean.shape[1]):
    x = v4_resp_mean[:,cell]
    nstim = x.shape[0]
    r_list = []

    sim_resp = np.random.poisson(x, size=(n_noise_resample, n_trials, nstim))
    r = [np.abs(np.corrcoef(resp, x)[0, 1]) for resp in sim_resp.mean(1)]
    apc_m_cor_list.append(np.mean(r))
    apc_sd_cor_list.append(np.std(r))
    
    
#%%
import apc_model_fit as ac

nm = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370.nc'
da = xr.open_dataset(top_dir + '/data/responses/'+ nm)['resp'].squeeze().load()
center_pos = np.round(len(da.coords['x'])/2.).astype(int)
da_0 = da.sel(x=da.coords['x'][center_pos])

fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()

apc_fit_v4 = ac.cor_resp_to_model(v4_resp_mean.chunk({'shapes': 370}), 
                                  dmod.chunk({}), 
                                  fit_over_dims=None, 
                                  prov_commit=False)
apc_fit_alex = ac.cor_resp_to_model(da_0.chunk({'shapes': 370}), 
                                  dmod.chunk({}), 
                                  fit_over_dims=None, 
                                  prov_commit=False)
#%%
import d_net_analysis as na
k = na.kurtosis_da(da_0)
non_k_var = (k>1) * (k<42) 
apc_fit_alex_k_filt = apc_fit_alex[non_k_var]

#%%
from scipy.stats import linregress
r_thresh = 0.70
fit_mod = []
mod_r = []
for i, fit in enumerate(apc_fit_v4):
    mod = dmod[:, int(fit.models.values)]
    dat = v4_resp_mean[:, i].values
    slope, inter, r, p, sterr = linregress(mod.values, dat)
    mod_r.append(r)
    fit_mod.append(slope*mod + inter)

n_noise_resample = 100
n_trials = 5  
perf_apc_m_cor_list = []
corresponding_v4_ind = []
for i, a_fit_mod, an_r in zip(range(len(fit_mod)), fit_mod, mod_r):
    if an_r>r_thresh:
        x = a_fit_mod
        if sum(x<0)>0:
            x = x - x.min()
        nstim = x.shape[0]
    
        sim_resp = np.random.poisson(x, size=(n_noise_resample, n_trials, nstim))
        r = [np.abs(np.corrcoef(resp, x)[0, 1]) 
            for resp in sim_resp.mean(1)]
        perf_apc_m_cor_list.append(np.mean(r))
        corresponding_v4_ind.append(i)
#%%    


plt.figure()
apc_fit_v4.plot.hist(cumulative=True, histtype='step', bins= 100, normed=True)
(apc_fit_v4 + (1-np.array(apc_m_cor_list))).plot.hist(cumulative=True, histtype='step', bins= 100, normed=True)
apc_fit_alex_k_filt.plot.hist(cumulative=True, histtype='step', bins= 1000, normed=True)
plt.xlim(0,1)
plt.yticks([0,0.25,0.5,0.75,1])
plt.legend(['V4-APC r', 'V4-APC exp. r', 'Alex-APC r'], loc=2)
plt.grid()
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'+str('3_')+ 
            'APC_noise_adj.pdf', bbox_inches='tight')

#%%
plt.figure()

plt.scatter(apc_fit_v4, (1-np.array(apc_m_cor_list)))
#plt.scatter([1,]*len(perf_apc_m_cor_list), (1-np.array(perf_apc_m_cor_list)))
#plt.legend(['V4', 'APC fit to V4'])

#for i, ind in enumerate(corresponding_v4_ind):
#    plt.plot([apc_fit_v4[ind], 1], 
#             [(1-np.array(apc_m_cor_list))[ind], 
#              (1-np.array(perf_apc_m_cor_list))[i]], color='k')
plt.title('APC: V4 Correlation to Model')
plt.xlabel('Fit to APC model')
plt.ylabel('Simulated Drop in Correlation')
plt.xlim(0,1)
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'+str('2_') + 
            'APC_noise_sim_predicts_low_cor_no_perf.pdf', bbox_inches='tight')
#%%
plt.figure()
plt.hist(apc_m_cor_list)
plt.title('APC: Noise Correlation to Perfect Model After Noise')
plt.xlabel('Correlation')
plt.ylabel('Cell Count')
plt.xlim(0,1)
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'+str('1_')+ 
            'APC_perfect_model_noise_sim.pdf', bbox_inches='tight')

#%%
#ti distribution
n_noise_resample = 100 #how many times to repeat the simulation
n_trials = 5
ti_m_cor_list = []
ti_sd_cor_list = []
v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()
for i in range(v4_resp_ti.shape[0]):
    x = v4_resp_ti[i].dropna('x', how='all').dropna('shapes', how='all').values
    x = x.ravel()
    nstim = len(x)
    sim_resp = np.random.poisson(x, size=(n_noise_resample, n_trials, nstim)) 
    r = [np.abs(np.corrcoef(resp, x)[0, 1]) for resp in sim_resp.mean(1)]

    ti_m_cor_list.append(np.mean(r))
    ti_sd_cor_list.append(np.std(r))       
#%%
v4_ti = []
for i in range(v4_resp_ti.shape[0]):
    x = v4_resp_ti[i].dropna('x', how='all').dropna('shapes', how='all')
    x = x.transpose('shapes', 'x')
    v4_ti.append(na.norm_cov(x.values))
#%%
alex_ti = na.ti_in_rf(da)
k = na.kurtosis_da(da)
non_k_var = (k[1]<42) * (k[0]<6)
#%%
alex_ti = alex_ti[non_k_var]

#%%
#get average rf profile.
#get strongest response.
#replicate response over rf
n_trials = 5
perf_ti_m_cor_list = [] 
v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()
for i in range(v4_resp_ti.shape[0]):
    x = v4_resp_ti[i].dropna('x', how='all').dropna('shapes', how='all').values
    rf = np.sum(x**2, 1, keepdims=True)**0.5
    rf_cent = x[rf.argmax(), :]
    perf_ti = np.tile(rf_cent, (x.shape[0], 1))
    perf_ti = (perf_ti / rf.max()) * rf
    perf_ti = perf_ti.ravel()    
    nstim = len(perf_ti)

    sim_resp = np.random.poisson(perf_ti, size=(n_noise_resample, n_trials, nstim)) 
    r = [na.norm_cov(resp.reshape(x.shape).T) for resp in sim_resp.mean(1)]
    perf_ti_m_cor_list.append(np.mean(r))

#%%
plt.figure()
plt.title('Comparing Perfect TI noise to Original Resp noise')
plt.scatter(v4_ti, 1- np.array(perf_ti_m_cor_list))
plt.xlabel('TI V4')
plt.ylabel('Simulated Drop from Perfect TI with Poisson noise')
plt.xlim(0,1);plt.ylim(0,1)
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'+str('1_')+ 
            'TI_perfect_model_noise_vs_V4ti.pdf', bbox_inches='tight')
#%%
plt.figure()
plt.hist(ti_m_cor_list)
plt.title('TI: Perfect TI After Noise')
plt.xlabel('Correlation')
plt.ylabel('Cell Count')
plt.xlim(0, 1)
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'+str('2_')+ 
            'TI_perfect_model_noise_sim.pdf', bbox_inches='tight')

#%%
plt.figure()

plt.scatter(v4_ti, (1-np.array(ti_m_cor_list)))
plt.title('TI: V4 Correlation to Model')
plt.xlabel('TI metric')
plt.ylabel('Simulated Drop in Correlation')
plt.ylim(0,.30);plt.xlim(0,1)
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'+str('3_')+ 
            'TI_noise_sim_predicts_low_cor.pdf', bbox_inches='tight')

#%%%
plt.figure()
plt.hist(v4_ti, cumulative=True, histtype='step', bins= 100, normed=True)
plt.hist(v4_ti + (1-np.array(perf_ti_m_cor_list)), cumulative=True, histtype='step', bins= 100, normed=True)
plt.hist(alex_ti.dropna('unit').values, cumulative=True, histtype='step', bins=100, normed=True)
plt.xlim(0,1)
plt.yticks([0,0.25,0.5,0.75,1])
plt.legend(['V4-TI r', 'V4-TI max r', 'Alex-TI r'], loc=2)
plt.grid()
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'+str('4_')+ 
            'TI_noise_adj.pdf', bbox_inches='tight')




#%%
'''
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
'''









