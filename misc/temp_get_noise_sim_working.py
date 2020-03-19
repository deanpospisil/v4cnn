# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:51:04 2017

@author: deanpospisil
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as  l
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')
import xarray as xr
import pandas as pd

v4_resp_mean = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc')['resp'].load()
v4_resp_trials = xr.open_dataset(top_dir + 'data/responses/apc_orig/apc370_with_trials.nc')
v4_resp_trials = v4_resp_trials['resp'].load().transpose('shapes','unit','trials')
v4_resp_trials = v4_resp_trials.loc[v4_resp_mean.coords['shapes'].values, :, :]
v4_resp_mean = v4_resp_trials.mean('trials')
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
    #solve for scaling of a to the opposite side for beta
    a = (mag_b * np.tan(beta))/mag_e
    shifted_vector = a * e + b # now shift b with a*e
    return shifted_vector

x = v4_resp_mean[:,22]
nstim = x.shape[0]
correlations = np.linspace(0, 1, 11)
n_model_resample = 50
n_noise_resample = 50
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
m_cor = r_list.mean(1).squeeze()

plt.subplot(121)
plt.errorbar(correlations, m_cor, yerr=np.std(r_list,1))
plt.xlabel('beginning correlation')
plt.ylabel('average correlation with noise')
plt.axis('square')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.plot([0,1],[0,1])

plt.subplot(122)
plt.plot(correlations, correlations-m_cor)
plt.xlabel('beginning correlation')
plt.ylabel('beginning correlation - correlation with noise')

plt.tight_layout()

#why is it a line?
#all of my neural correlations should be less than the max of m_cor











