# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:26:54 2018

@author: deanpospisil
"""

import numpy as np
import matplotlib.pyplot as plt

n = 10
trials = 10000
lam = 50
rho = 0.33

lam_c = lam*rho
lam_i = lam - lam_c
#lam_c = 1000
#lam_i = 20

mean_r = lam
var_r = (n**-2)*(lam*n  + (n**2-n)*lam_c)

x = np.random.poisson([lam_i]*n, size=(1, trials,n))
c = np.random.poisson(lam_c, size=(1, trials, 1))

x = x+c


w = 1/np.double(n)

r = np.sum(w*x, -1)

var = np.var(r,1, ddof=1)
mu = np.mean(r,1)
est_rho = np.corrcoef(np.squeeze(x.T))[0,1]

print('est_var =' + str(var)+' True Var =' + str(var_r))
print('est_mean =' +str(mu)+' True Mean =' + str(mean_r))
print('est_rho =' +str(est_rho)+' True rho =' + str(rho))
#%%
n = 10
trials = 10000
lam = np.array(np.arange(1, 100, 5))
rho = 0.3

lam_cs = lam*rho
lam_is = lam - lam_cs

mean_rs = lam
var_rs = (n**-2)*(lam*n  + (n**2-n)*lam_cs)

x = np.concatenate([np.random.poisson([lam_i]*n, size=(1, trials,n)) 
                for lam_i in lam_is])
c = np.concatenate([np.random.poisson(lam_c, size=(1, trials, 1)) 
                    for lam_c in lam_cs])

x = x+c

w = 1/np.double(n)

r = np.sum(w*x, -1)

var = np.var(r, 1, ddof=1)
mu = np.mean(r, 1)

plt.figure(figsize=(4,4))
plt.scatter(mu,var)
plt.plot(mean_rs, var_rs, color='r')
plt.xlabel('Mean')
plt.ylabel('Variance')

plt.legend(['Theory', 'Simulation'])
plt.xlim(0, max(mean_rs));plt.ylim(0, max(mean_rs))

#%%
def cor_poisson_sim(n, trials, lam, rho, b=0):
    #takes array of lams, with each simulates the integration of n poisson
    #random variables with correlation=rho for each by adding a common source
    w = 1/np.double(n)
    lam_cs = lam*rho
    lam_is = lam - lam_cs
    theory_mean = lam
    theory_var = (n**-2)*(lam*n  + (n**2-n)*lam_cs)
    
    x = np.concatenate([np.random.poisson([lam_i]*n, size=(1, trials,n)) 
                for lam_i in lam_is])
    c = np.concatenate([np.random.poisson(lam_c, size=(1, trials, 1)) 
                        for lam_c in lam_cs])
    x = x+c
    r = np.sum(w*x, -1)#integrate over units
    
    est_mu = np.mean(r, 1)
    est_var = np.var(r, 1, ddof=1)
    return est_mu, est_var, theory_mean, theory_var
def cor_poisson_theory(n, lam, rho, b=0):
    #takes array of lams, with each simulates the integration of n poisson
    #random variables with correlation=rho for each by adding a common source
    lam_cs = lam*rho
    theory_mean = lam-b
    theory_var = (n**-2)*(lam*n  + (n**2-n)*lam_cs)
    
    return theory_mean, theory_var

n = 10
trials = 1000
lam = np.array(np.arange(1, 100, 5))
rhos = np.arange(0, 1.2, 0.2)
rho_theory = [cor_poisson_theory(n, lam, rho) for rho in rhos]
plt.figure(figsize=(4,4))

for res in rho_theory:
    plt.plot(res[0], res[1])

plt.legend(rhos, title='rho')
plt.title('n = '+str(n)+'\nSlope = rho + (1/n - rho/n)');plt.xlabel('Mean');plt.ylabel('Variance')
 
n = 3
rho_theory = [cor_poisson_theory(n, lam, rho) for rho in rhos]
plt.figure(figsize=(4,4))

for res in rho_theory:
    plt.plot(res[0], res[1])

plt.legend(rhos, title='rho')
plt.title('n = '+str(n)+'\nSlope = rho + (1/n - rho/n)');plt.xlabel('Mean');plt.ylabel('Variance')
 
   

#%%
n = 10
trials = 1000
lam = np.array(np.arange(1, 200, 5))
rhos = np.arange(0, 1.2, 0.2)
bs = range(0,130,30)
b_theory = [cor_poisson_theory(n, lam, rho=0.3, b=b) for b in bs ]
plt.figure(figsize=(4,4))

for res in b_theory:
    plt.plot(res[0], res[1])

plt.legend(bs, title='b')
plt.title('n = '+str(n)+'\nSlope = rho + (1/n - rho/n)');plt.xlabel('Mean');plt.ylabel('Variance')
plt.plot([0,100], [0,100], color='k')
plt.xlim(-10, 100);plt.ylim(-10, 100);


#    plt.gca().set_yscale('log')
#    plt.gca().set_xscale('log')




