# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:43:58 2017

@author: deanpospisil
"""
import pandas as pd
def norm_av_cov(unit, return_num_den=False):
    unit = unit.transpose('shapes','x')
    unit = unit.dropna('x', 'all').dropna('shapes', 'all').values
    cov = np.dot(unit.T, unit)
    cov[np.diag_indices_from(cov)] = 0
    numerator = np.sum(np.triu(cov))
    vlength = np.linalg.norm(unit, axis=0)
    max_cov = np.outer(vlength.T, vlength)
    max_cov[np.diag_indices_from(max_cov)] = 0
    denominator= np.sum(np.triu(max_cov))
    if return_num_den:
        return numerator, denominator
    else:
        return numerator/denominator
    
top_dir = '/Users/deanpospisil/Desktop/modules/v4cnn/'
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
fn = top_dir +'data/responses/v4_ti_resp.nc'
v4 = xr.open_dataset(fn)['resp'].load()
t = [norm_av_cov(v4[i] - v4[i].mean('shapes')) for i in range(80)]


max_mean_ind = v4.max('shapes').argmax('x')
trials = 5
n_repeats = 40
av_r = []
for unit, pos in zip(v4, max_mean_ind):
    resp = unit[int(pos.values)].dropna('shapes').values
    a = np.random.poisson(lam=resp, size=(n_repeats, trials, len(resp)))
    repeats = np.mean(a, 1)
    
    resp = resp - np.mean(resp)
    resp = resp / sum(resp**2)**0.5
    
    repeats = repeats - np.mean(repeats, 1, keepdims=True)
    repeats = repeats / np.sum(repeats**2, 1, keepdims=True)**0.5

    av_r.append(np.mean(np.dot(resp, repeats.T)))
   
plt.scatter(av_r, t)  
plt.xlabel('Average R Perfect Model')
plt.ylabel('Translation Invariance')

plt.figure()
plt.scatter(v4.max(['shapes','x'])-v4.min(['shapes','x']), t)
plt.xlabel('Neuron range')
plt.ylabel('Translation Invariance')

plt.figure()
plt.scatter(v4.max(['shapes', 'x'])-v4.min(['shapes', 'x']), av_r)
plt.ylabel('Average R Perfect Model')
plt.xlabel('Neuron range')

#%%
cat = pd.read_csv(top_dir + 'data/responses/PositionData_Yasmine/TXT_category', delimiter=' ')
cat = cat['c'].values

t = [norm_av_cov(v4[i] - v4[i].mean('shapes')) for i in range(80)]
max_mean_ind = v4.mean('shapes').argmax('x')
snr = []
for cell, pos in enumerate(max_mean_ind.values):
    snr.append(v4[cell].var('shapes')[pos].values**0.5)

colors = []
cav_pref = []
vex_pref = []
vex_pref_snr = []
cav_pref_snr = []

for i, a_cat in enumerate(cat):
   if a_cat == 1:
       colors.append('c')
       vex_pref.append(t[i])
       vex_pref_snr.append(snr[i])

   else:
       colors.append('orange')
       cav_pref.append(t[i])
       cav_pref_snr.append(snr[i])

plt.scatter(snr, t, c=colors)
plt.xlabel('SD of RF center responses')
plt.ylabel('Translation invariance')
plt.title('Blue prefer convex.')

plt.savefig(top_dir + 'ti_vs_sd_for_anitha.pdf')

plt.figure(figsize=(4,8))
plt.subplot(211)
plt.title('Cumulative')
plt.hist(vex_pref, cumulative=True, histtype='step', bins=1000)
plt.hist(cav_pref, cumulative=True, histtype='step', bins=1000)
plt.legend(['convex', 'concave'], loc=2)

plt.subplot(212)
plt.title('Histogram')
plt.hist(vex_pref, cumulative=False, histtype='step', bins=10)
plt.hist(cav_pref, cumulative=False, histtype='step', bins=10)
plt.legend(['convex', 'concave'], loc=2)
plt.xlabel('SNR Metric')
plt.savefig(top_dir + 'ti_vex_vs_cav_for_anitha.pdf')





#%%
inds = np.argsort(snr)
inds = [51, 70, 59, ]
for ind in inds[:]:
    plt.figure()
    v4.isel(unit=ind).plot()
    plt.title('TI = '+ str(np.round(t[ind],2)) +
            ' SD = ' + str(np.round(snr[ind],2)) + 
            ' Prefer convex: ' + str(cat[ind]==1)+
            ' cell:' + str(ind))
    plt.savefig(top_dir +'cell:' + str(ind))
    
#%%
for i in range(80):  
    unit = v4[i]
    unit = unit.transpose('shapes','x')
    unit = unit.dropna('x', 'all').dropna('shapes', 'all')
    p = unit.to_pandas()
    p.to_csv(top_dir + 'data/responses/ti_csvs/' + str(i))    


