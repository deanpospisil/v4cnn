# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:52:03 2017

@author: deanpospisil
"""
import numpy as np
#first we perform an experiment
n = 5 #trials
s = 10 #stimuli
n_exps = 10000 #number simulations
sigma = np.array([1,]*s) # the variance of each trial

#each trial for a stimuli has the same expectation
neuron_mu = np.cos(np.linspace(0, np.pi, s)) + 1 #s

#calc var_sum
sigma_R = (sigma/n) #averaging trials reduces variance.
var_sum = np.sum(sigma_R)

#sum of mu^2                    
mu_sqrd_sum = np.sum(neuron_mu**2)

#cross products of mu's
cross_prod = ((np.expand_dims(neuron_mu, 0) * np.expand_dims(neuron_mu, 1)))
cross_prod_sum = np.sum(cross_prod) - np.sum(np.trace(cross_prod))

print('Sum of variances: '+ str(var_sum))
print('Sum of cross product means: ' + str(cross_prod_sum))
print('Sum of mean squared: ' + str(mu_sqrd_sum))

#we then perform our experiments
r = np.random.normal(loc=neuron_mu, scale=np.sqrt(sigma), 
                     size=(n_exps, n, s)) # nx n s

#calc var_sum
est_neuron_sig = np.var(r, 1, ddof=1) #nx s
est_sigma_R = est_neuron_sig/n #averaging trials reduces variance.
est_var_sum = np.sum(est_sigma_R, 1)
                     
 #sum of mu^2                    
est_neuron_sig = np.var(r, 1, ddof=1) #nx s
est_neuron_mu2 = np.mean(r**2, 1)# nx s
est_neuron_mu2 -= est_neuron_sig
est_mu_sqrd_sum = est_neuron_mu2.sum(1)

#cross products of mu's
est_mu = np.mean(r, 1)
est_cross_prod = (np.expand_dims(est_mu, 1) * np.expand_dims(est_mu, 2))#outer product of mus
est_cross_prod_sum = np.sum(est_cross_prod, axis=(1,2)) - np.trace(est_cross_prod, axis1=1, axis2=2)

#seems to work
print('Average estimated sum of variances: ' + str(np.round(np.mean(est_var_sum),3)))
print('Average estimated sum of cross product means: ' + str(np.round(np.mean(est_cross_prod_sum),3)))
print('Average estimated sum of mean squared: ' + str(np.round(np.mean(est_mu_sqrd_sum),3)))

#%%
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import xarray as xr 
import xarray as xr
cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',]
da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
da = da.sel(unit=slice(0, None, 1)).squeeze()
middle = np.round(len(da.coords['x'])/2.).astype(int)
da_0 = da.sel(x=da.coords['x'][middle])
