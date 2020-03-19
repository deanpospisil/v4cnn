# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 05:33:08 2020

@author: deanpospisil
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def s2_hat_obs_n2n(x, y):
    s2_hat = (np.mean(np.var(y, -2, ddof=1, keepdims=True), -1, keepdims=True) +
          np.mean(np.var(x, -2, ddof=1, keepdims=True), -1, keepdims=True))/2.
    return s2_hat

def mu2_hat_obs_n2n(x, y):
    
    xm = np.mean(x, -2, keepdims=True)
    x_ms = xm - np.mean(xm, -1, keepdims=True)
    
    ym = np.mean(y, -2, keepdims=True)
    y_ms = ym - np.mean(ym, -1, keepdims=True)
    
    
    mu2x = np.sum(x_ms**2, -1, keepdims=True)
    mu2y = np.sum(y_ms**2, -1, keepdims=True)
    
    return mu2x, mu2y

def r2c_n2n(x,y):
    """approximately unbiased estimator of R^2 between the expected values. 
        of the columns of x and y. Assumes x and y have equal variance across 
        trials and observations.
    Parameters
    ----------
    x : numpy.ndarray
        n trials by m observations array
    y : numpy.ndarray
        n trials by m observations array
  
    Returns
    -------
    r2c : an estimate of the r2 between the expected values
    r2 : the fraction explained variance between the mean observations
    --------
    """
    
    n, m = np.shape(y)[-2:]
    
    sig2_hat = s2_hat_obs_n2n(x, y)
    
    x = np.mean(x, -2, keepdims=True)
    x_ms = x - np.mean(x, -1, keepdims=True)
    
    y = np.mean(y, -2, keepdims=True)
    y_ms = y - np.mean(y, -1, keepdims=True)
    
    xy2 = np.sum((x_ms*y_ms), -1, keepdims=True)**2
    x2 = np.sum(x_ms**2, -1, keepdims=True)
    y2 = np.sum(y_ms**2, -1, keepdims=True)
    x2y2 = x2*y2
    
    ub_xy2 = xy2 - (sig2_hat/n*(x2 + y2 - (m-1)*sig2_hat/n))
    ub_x2y2 = x2y2 - (m-1)*sig2_hat/n*(x2 + y2 - (m-1)*sig2_hat/n)
    
    r2c = ub_xy2/ub_x2y2
    
    return r2c, xy2/x2y2

n_sims = 100
m_stim = 30
n_trials = 4
n_neuron_pairs = 60

theta = np.linspace(0, 2*np.pi - 2*np.pi/m_stim, m_stim)[np.newaxis, :, 
                   np.newaxis,np.newaxis,np.newaxis]
pref_theta = np.random.uniform(0, np.pi/8, (n_sims, 1, 1, n_neuron_pairs, 2))

amp = np.random.uniform(5, 10, (n_sims, 1, 1, n_neuron_pairs, 2))

tuning_curves = (np.cos(theta + pref_theta) + 1 )*amp/2 + 5

resp = np.random.poisson(tuning_curves, (n_sims, m_stim, n_trials, n_neuron_pairs, 2))


#plt.plot(tuning_curves[1,:,0,0])


def norm(x, dim):
    x = x - x.mean(dim)
    x = x/(x**2).sum(dim)**0.5
    return x
    
    
def da_cor(x, y, dim='stim'):
    x = norm(x, dim)
    y = norm(y, dim)
    r = (x*y).sum(dim)
    return r
    
    

tc = xr.DataArray(tuning_curves, dims=['sim', 'stim', 'trial' ,'rec', 'pair'],
                    coords=[range(a) for a in tuning_curves.shape]).squeeze()

resp = xr.DataArray(resp, dims=['sim', 'stim', 'trial', 'rec', 'pair'],
                    coords=[range(a) for a in resp.shape])

r2_true = da_cor(tc.sel(pair=0), tc.sel(pair=1), dim='stim')**2
r2_true.plot.hist(bins=100)

r2_naive = da_cor(resp.sel(pair=0).mean('trial'), 
                  resp.sel(pair=1).mean('trial'), dim='stim')**2
plt.figure()
r2_naive.plot.hist(bins=100)

resp = resp.transpose('sim','rec', 'pair', 'trial', 'stim')

r2c, r2_naive = r2c_n2n(resp.sel(pair=0).values**0.5, 
                        resp.sel(pair=1).values**0.5)
#r2c[r2c>1] = 1
da_r2c = r2_true.copy(deep=True)
da_r2c[...] = r2c.squeeze() 

plt.figure()
da_r2c.plot.hist(bins=100)

mu2y, mu2x = mu2_hat_obs_n2n(resp.sel(pair=0).values**0.5, 
                             resp.sel(pair=1).values**0.5)

dyn = ((mu2x*mu2y)**0.5).squeeze()

plt.figure()

plt.scatter(dyn.ravel()[::5], r2c.ravel()[::5])

plt.figure()
plt.scatter(dyn.ravel()[::5], r2_naive.ravel()[::5])



