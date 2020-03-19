# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:09:02 2018

@author: deanpospisil
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
#%%

data_dir = '/Users/deanpospisil/Desktop/modules/v4cnn/data/responses/'
da = xr.open_dataset(data_dir + 'taek_tex_shape').load()['resp']
#%%
def xr_proj(x, y, dim):
    x_2 = (x**2).sum(dim)
    cov = (x*y).sum(dim)
    beta = cov/x_2
    return beta

#windows respect causality, i.e. are to the left of time. so 50 is 0 to 50 
stim_start = 50
stim_end = 400

for stim_end in range(100, 750,100):
    stim_dur = (stim_end - stim_start + 50)/1000
    
    time = slice(stim_start, stim_end)
    da_t = da.sel(time=time).sum('time', skipna=False)
     
    m = da_t.mean('trial', skipna=True).dropna('stim', how='all')
    v = da_t.var('trial', skipna=True, ddof=1).dropna('stim', how='all')
    
    b = xr_proj(m, v, ['stim',])
    dyn = np.sqrt(da_t).mean('trial', skipna=True
             ).groupby('stim_type').var('stim', skipna=True, ddof=1)
    dt = dyn.sel(stim_type='st')
    ds = dyn.sel(stim_type='s')
    dif_dyn = dt - ds
    #plt.figure()
    sel = (b<10)*((ds>0) + (dt>0))
    plt.figure()
    r = np.corrcoef(b[sel], dif_dyn[sel])[0,1]**2
    plt.scatter(b[sel], dif_dyn[sel])
    plt.title('r^2 =' +str(np.round(r,2)) + ' stim_time = [+50, +' +str(stim_end)+']')
    plt.xlabel('Mean to Variance ratio');plt.ylabel('T-S')
#%%
for cell_ind in b.argsort()[::-1][:5]:
    plt.figure()
    m1 = m.sel(unit=cell_ind)
    v1 = v.sel(unit=cell_ind)
    ind = (m1>0)*(v1>0)
    
    plt.scatter(m1[ind], v1[ind]);
    plt.xlim([1,50]);
    plt.ylim([1,50]);plt.axis('equal');
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.plot([1,100],[1,100])
    
    np.corrcoef(np.log(m1),np.log(v1))

#%%
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color='r')
    
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pandas as pd


stim_start = 50
stim_end = 700


stim_dur = (stim_end - stim_start + 50)/1000

time = slice(stim_start, stim_end)
da_t = da.sel(time=time).sum('time', skipna=False)
 
m = da_t.mean('trial', skipna=True).dropna('stim', how='all')
v = da_t.var('trial', skipna=True, ddof=1).dropna('stim', how='all')


b = xr_proj(m, v, ['stim',])
dyn = np.sqrt(da_t).mean('trial', skipna=True
         ).groupby('stim_type').var('stim', skipna=True, ddof=1)
dt = dyn.sel(stim_type='st')
ds = dyn.sel(stim_type='s')
dif_dyn = dt - ds
#plt.figure()
sel = (b<5)*((ds>0.25) + (dt>0.25))
r = np.corrcoef(b[sel], dif_dyn[sel])[0,1]**2
plt.scatter(b[sel], dif_dyn[sel])

coefs = []
for var, mean in zip(m, v):
    ind = (var>0)*(mean>0)
    df = pd.DataFrame(np.array([np.log(var.values[ind]), np.log(mean.values[ind])]).T, columns=['mean', 'var'])
    mod = (ols('var ~ mean ', data=df).fit())
    
    df = pd.DataFrame(np.array([var.values[ind], mean.values[ind]]).T, columns=['mean', 'var'])
    modlin = (ols('var ~ mean -1', data=df).fit())
    coefs.append([mod.params[1], np.exp(mod.params[0]), modlin.params[0]])

    #df.plot.scatter('mean', 'var')
    #abline(mod.params[1], mod.params[0])

    #plt.title('exp= '+ str(np.round(mod.params[1],2)) + ' slope= ' +  str(np.exp(np.round(mod.params[0],2))))
    #plt.xlim([-4,5]);plt.ylim([-4,5]);
    #plt.plot([-4,4], [-4,4])
    
coefs = np.array(coefs)



#%%
#plt.plot(coefs);plt.legend(['exp', 'slope'])
#sel = (np.abs(dif_dyn)>0.02)*(np.abs(b)<9)
#sel = (np.abs(b)<2)
sel = (coefs[:,0]<1.2)*(coefs[:,0]>0.7)

#sel = (ds.values>.1)*(dt.values>.1)
#sel = (coefs[:,2]<1.3)

#sel = [True,]*len(dif_dyn)

df2 = pd.DataFrame(np.array([coefs[:,0][sel], coefs[:,1][sel], coefs[:,2][sel], dif_dyn.values[sel]]).T, 
                            columns=['exp', 'slope_exp',  'slope_lin',  'tex'])

mod = (ols('tex ~ slope_lin', data=df2).fit())
print(mod.summary2())
#plt.figure(figsize=(10,10))
df2.plot.scatter('slope_lin', 'tex', s=10, figsize=(5,5))
#for i, exp in enumerate(np.round(df2['d'].values,2).astype(str)):
#    plt.text(df2['exp'][i], df2['slope_exp'][i],  exp, fontsize=9)
    
plt.xlabel('Variance to Mean Slope')
plt.ylabel('Texture - Form Dynamic range')

#%%
#dat = np.array([coefs[:,0], coefs[:,1], coefs[:,2], dif_dyn.values, dt.values, ds.values]).T
#df2 = pd.DataFrame(dat, columns=['exp', 'slope_exp',  'slope_lin',  'tex-shape', 'tex', 'shape'])
#from glue import qglue
#app = qglue(xyz=df2)
#%%

da = xr.open_dataset(data_dir + 'taek_tex_shape').load()['resp']

stim_dur = (stim_end - stim_start + 50)/1000

time = slice(stim_start, stim_end)
da_t = da.sel(time=time).sum('time', skipna=False)
 
mt = da_t.sel(stim_type='st').mean('trial', skipna=True).dropna('stim', how='all')
vt = da_t.sel(stim_type='st').var('trial', skipna=True, ddof=1).dropna('stim', how='all')

ms = da_t.mean('trial', skipna=True).dropna('stim', how='all')
vs = da_t.sel(stim_type='s').var('trial', skipna=True, ddof=1).dropna('stim', how='all')



bt = xr_proj(mt, vt, ['stim',])
bs = xr_proj(ms, vs, ['stim',])

#%%

r = (da_t.mean('trial')*da_t.var('trial'))/(da_t.mean('trial')**2)


#%%
avs = []
for var, mean in zip(m, v):
    ind = (var>20)*(mean>20)
    var = var[ind]
    mean = mean[ind]
    disp = (var/mean).dropna('stim')
    avs.append(disp.mean('stim').values)
avs = np.array(avs)
    

df2 = pd.DataFrame(np.array([avs,  dif_dyn.values]).T, 
                            columns=['slope_lin', 'tex'])

mod = (ols('tex ~ slope_lin', data=df2).fit())
print(mod.summary2())
#plt.figure(figsize=(10,10))
df2.plot.scatter('tex', 'slope_lin', s=10, figsize=(5,5))