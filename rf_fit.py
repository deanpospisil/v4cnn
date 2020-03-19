# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:04:21 2019

@author: deanpospisil
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor
l
This is a temporary script file.
"""


import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt, seaborn as sns
from scipy import  optimize as op
#%%
# set random seed for reproducibility
np.random.seed(12345)

size=10
def gaus(x, mu=0, sig=1, a=10):
    e = (-(x-mu)**2)/(2*sig**2)
    y = a*np.exp(e)
    return y
  #%% 
size=3
x = np.linspace(-2,2, size)
t_mu, t_sig, t_a, t_nsig = [1, 3, 10, 1]
y = gaus(x, mu=t_mu, sig=t_sig, a=t_a)


Y = np.random.poisson(y)*t_nsig

plt.plot(x,Y)


basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    mu = pm.Normal('mu', mu=0, sd=10)
    sig = pm.HalfNormal('sig', sd=10)
    a = pm.Normal('a', mu=1, sd=10)
    nsig = pm.HalfNormal('nsig', sd=10)
    
    
    rf = gaus(x, mu=mu, sig=sig, a=a)
    
    Y_obs = pm.NegativeBinomial('Y_obs', mu=rf, alpha=nsig,  observed=Y)
 
with basic_model:
    # draw 500 posterior samples
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, tune=200, chains=2)
    


#%%
df = pm.summary(trace)
f_mu, f_a, f_sig, f_nsig = df['mean']

new_x = np.linspace(-10, 10, size*10)

fit_y = gaus(new_x, mu=f_mu, sig= f_sig, a=f_a)
y = gaus(new_x, mu=t_mu, sig= t_sig, a=t_a)
plt.plot(new_x, fit_y)
plt.plot(new_x, y);
plt.plot(x,Y)
plt.legend(['fit', 'true', 'data'])
pm.summary(trace).round(2)

#%%
#first choose guess x's across the SD's of the guess
#choose the expected y at that x
#append it to the data
#get the difference in confidence interval with and with out that point
#choose the biggest average percentage reduction in CI.
# draw a sample and continue. 

ci = df['hpd_97.5'] - df['hpd_2.5']
#%%

import pandas as pd
x = np.linspace(-10,10, size)
t_mu, t_sig, t_a, t_nsig = [1, 3, 10, 1]
y = gaus(x, mu=t_mu, sig=t_sig, a=t_a)


Y = y + np.random.randn(size)*t_nsig

popt, pcov = op.curve_fit(gaus, x, Y)

f_mu, f_sig, f_a,  = popt 

perr = np.sqrt(np.diag(pcov))

fit_y = gaus(new_x, *popt)
y = gaus(new_x, mu=t_mu, sig= t_sig, a=t_a)
plt.plot(new_x, fit_y)
plt.plot(new_x, y);
plt.plot(x,Y)
plt.legend(['fit', 'true', 'data'])
dat = np.zeros((3,2))
labels = ['mu', 'sig', 'a']
dat[:,0] = popt
dat[:,1] = perr

df2 = pd.DataFrame(dat)
df2.index = labels
df2.columns = ['mean', 'sd']

print(pm.summary(trace).round(2))
print(df2)

#%%

orig_errs = perr
test_x = np.linspace(-10, 10, 100)
test_y = gaus(test_x, mu=f_mu, sig=f_sig, a=f_a)
test_Y = np.concatenate((Y,np.array([0,])))
test_X = np.concatenate((x, np.array([0,])))
errs = []

for a_x, a_y in zip(test_x, test_y):
    test_Y[-1] = a_y
    test_X[-1] = a_x
    
    popt, pcov = op.curve_fit(gaus, test_X, test_Y)
    f_mu, f_sig, f_a  = popt 
    perr = np.sqrt(np.diag(pcov))
    errs.append(perr)
    
errs = np.array(errs/perr)
errs_df = pd.DataFrame(errs,index=test_x, columns=labels)
errs_df.plot();
plt.ylabel('New SD / Original SD');
plt.xlabel('Degrees Visual Angle');
#%%
#def error_ratio(f, x, y, x_probes):
    #takes a function, predictors, responses, and predictors to try
    #evaluates probes at the expectation of the fit of f for
    #the SE relative to the original SE without the probe for all x_probes
    #puts these in a data frame
import inspect
import pandas as pd
f = gaus
x = np.linspace(-10,10, size)
t_mu, t_sig, t_a, t_nsig = [1, 3, 10, .25]
y = gaus(x, mu=t_mu, sig=t_sig, a=t_a)


Y = y + np.random.randn(size)*t_nsig



def error_ratio(f, x, y, x_probes):
    labels = list(inspect.signature(f).parameters.keys())[1:]

    popt, pcov = op.curve_fit(f, x, y)
    o_errs = np.sqrt(np.diag(pcov))
    
    y_probes = f(x_probes, *popt)
        
    y_new = np.concatenate((y,np.zeros(1)))
    x_new = np.concatenate((x,np.zeros(1)))
    
    new_errs = []
    for a_x, a_y in zip(x_probes, y_probes):
        y_new[-1] = a_y
        x_new[-1] = a_x
        
        popt, pcov = op.curve_fit(f, x_new, y_new)
        
        new_err = np.sqrt(np.diag(pcov))
        new_errs.append(new_err)
        
    rel_errs = np.array(new_errs)/o_errs
    
    d = pd.DataFrame(rel_errs, columns=labels, index=x_probes);
    return d

x_probes = np.linspace(-10,10,30)


d = error_ratio(f, x, y, x_probes)

d.plot();plt.ylabel('');
plt.ylabel('New SD / Original SD');
plt.xlabel('Degrees Visual Angle');
    
#plt.ylim(0,1)  
    
    
#%%
#sim
x_init = np.linspace(-10,10, 5)

x_probes = np.linspace(-10,10,30)

t_mu, t_sig, t_a, t_nsig = [0, 3, 10, 1]

def rf_sim(x, mu, sig, a, nsig):
    e_y = gaus(x, mu=mu, sig=sig, a=a)
    if np.shape(x)==():
        len_x = 1
    else:
        len_x = len(x)
    y = e_y + np.random.randn(len_x)*nsig
    return y

y_init = rf_sim(x_init, mu=t_mu, sig=t_sig, a=t_a, nsig=t_nsig)  


n_trials = 20 
columns = ['mu', 'sig', 'a', 'sd_mu', 'sd_sig', 'sd_a']
ad_results = pd.DataFrame(np.zeros((n_trials,len(popt)+len(pcov))), 
                       columns=columns)

y = y_init
x = x_init
for trial in range(n_trials):
    print(trial)
    er = error_ratio(f, x, y, x_probes)
    popt, pcov = op.curve_fit(f, x, y)
    ad_results.iloc[trial,:] = np.concatenate((popt,np.diag(pcov)**0.5))
    
    new_x = np.array([np.mean(er, 1).argmin(),])
    new_y = rf_sim(new_x, mu=t_mu, sig=t_sig, a=t_a, nsig=t_nsig)
    
    y = np.concatenate((y, new_y))
    x = np.concatenate((x, new_x))
    
rand_results = pd.DataFrame(np.zeros((n_trials,len(popt)+len(pcov))), 
                       columns=columns)

y = y_init
x = x_init
for trial in range(n_trials):
    print(trial)
    
    popt, pcov = op.curve_fit(f, x, y)
    rand_results.iloc[trial,:] = np.concatenate((popt,np.diag(pcov)**0.5))
    np.random.uniform(-10,10)
    new_x = np.array([np.random.uniform(-10, 10),])
    new_y = rf_sim(new_x, mu=t_mu, sig=t_sig, a=t_a, nsig=t_nsig)
    
    y = np.concatenate((y, new_y))
    x = np.concatenate((x, new_x))  


#%%

def ad_rf_sim(x_init, x_probes, mu=0, sig=5, a=10, nsig=.25, n_trials=20, rand=False, return_xy=False):
    columns = ['mu', 'sig', 'a', 'sd_mu', 'sd_sig', 'sd_a']
    ad_results = pd.DataFrame(np.zeros((n_trials, len(columns))), 
                       columns=columns)
    y = rf_sim(x_init, mu=mu, sig=sig, a=a, nsig=nsig)  
    x = x_init
    
    for trial in range(n_trials):
        print(trial)
        if rand:
            new_x = np.array([np.random.uniform(x_probes.min(),x_probes.max())])
        else:
            er = error_ratio(f, x, y, x_probes)
            new_x = np.array([np.mean(er, 1).idxmin(),])
            
        popt, pcov = op.curve_fit(f, x, y)
        ad_results.iloc[trial,:] = np.concatenate((popt, np.diag(pcov)**0.5))
        

        new_y = rf_sim(new_x, mu=mu, sig=sig, a=a, nsig=nsig)
        
        y = np.concatenate((y, new_y))
        x = np.concatenate((x, new_x))  
    if return_xy:
        return ad_results, x, y
    else:
        return ad_results
    
x_probes = np.linspace(-15,15,30)



#%%
n_sims = 30
n_trials = 4

rd_res = [ad_rf_sim(x_init, x_probes,n_trials=n_trials, rand=True).iloc[-1] for i in range(n_sims)]
rd_res = pd.concat(rd_res,1).T.iloc[:,:3]



#%%
ad_res = [ad_rf_sim(x_init, x_probes,n_trials=n_trials, rand=False).iloc[-1] for i in range(n_sims)]
ad_res = pd.concat(ad_res,1).T.iloc[:,:3]



#%%
ad_dif = ad_res.abs() - [0,5,10]
ad_var = (ad_dif**2).sum(0)

rd_dif = rd_res.abs() - [0, 5, 10]
rd_var = (rd_dif**2).sum(0)

print(ad_var)
print(rd_var)
#%%
rd_res_arr = np.array([res.values for res in rd_res])[..., np.newaxis]
rd_res_arr = np.concatenate(rd_res_arr, -1)
rd_res_acc = [0,5,10]
rd_res_sdm = rd_res_arr.mean(-1)
rd_res_m = pd.DataFrame(rd_res_m, columns=rd_res[0].columns)


#%%  
rd_res_m.iloc[:, 3:].plot()
ad_res_m.iloc[:, 3:].plot()



#%%
ad_res, x, y = ad_rf_sim(x_init, x_probes, return_xy=True, n_trials=n_trials)

plt.scatter(x,y)
#%%
rd_res, x, y = ad_rf_sim(x_init, x_probes,rand=True, return_xy=True, n_trials=n_trials)

plt.scatter(x,y)


#%%
#plt.scatter(x[5:], y[5:]) 

ad_results.iloc[3:, :3].abs().plot()
(ad_results.iloc[3:, 3:].abs()*4).plot()
plt.ylabel('Confidence Interval Length')
plt.xlabel('Trial')

#%%
#plt.scatter(x[5:], y[5:]) 

rand_results.iloc[3:, :3].abs().plot()
(rand_results.iloc[3:, 3:].abs()*4).plot()
plt.ylabel('Confidence Interval Length')
plt.xlabel('Trial')


#%%

(ad_results.iloc[3:, 3:].abs()*4).plot()
(rand_results.iloc[3:, 3:].abs()*4).plot()


#%% test if stimulatin at center and SD's is better than random for estimation.

t_mu, t_sig, t_a, t_nsig = [2, 3, 4, 0.25**0.5]
target = [t_mu, t_sig, t_a]

size = 60

x_grid = np.linspace(t_mu-t_sig*2, t_mu+t_sig*2, size)
x_opt = np.concatenate((-np.ones(10)*t_sig*1 + t_mu,
                        -np.ones(10)*t_sig + t_mu, 
                        np.zeros(20)+t_mu, 
                        t_mu+np.ones(10)*t_sig,
                        t_mu+np.ones(10)*t_sig*1))
nsim=500
sims = np.zeros((nsim, 2, 3))
for i in range(nsim):
    y_grid = np.random.poisson(gaus(x_grid, mu=t_mu, sig=t_sig, a=t_a))# + np.abs(np.random.randn(size))*t_nsig
    y_opt = np.random.poisson(gaus(x_opt, mu=t_mu, sig=t_sig, a=t_a))# + np.abs(np.random.randn(size))*t_nsig
    
    popt_grid, pcov = op.curve_fit(gaus, x_grid, y_grid, bounds=(0,np.inf))
    popt_opt, pcov = op.curve_fit(gaus, x_opt, y_opt, bounds=(0,np.inf))
    
    dif_grid = (np.abs(popt_grid) - target)**2
    dif_opt = (np.abs(popt_opt) - target)**2
    
    sims[i,0] = dif_grid 
    sims[i,1] = dif_opt
#%%
mse = np.mean(sims, 0)
sd_mse = np.sqrt(np.var(sims, 0)/nsim)

plt.errorbar(range(3), mse[0], yerr=sd_mse[0]);

plt.errorbar(range(3), mse[1], yerr=sd_mse[1]);

plt.legend(['grid', 'ad'])



#%%
plt.scatter(x_grid, y_grid)
plt.scatter(x_opt, y_opt)
