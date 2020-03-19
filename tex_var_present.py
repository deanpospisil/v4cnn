# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:36:55 2018

@author: deanpospisil
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols, rlm
import statsmodels.api as sm
import pandas as pd
save_dir = '/Users/deanpospisil/Desktop/writing/science/student symposium present/'
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color='b')
    
def expline(exp, slope):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.linspace(*axes.get_xlim())
    y_vals =  slope * x_vals ** exp
    plt.plot(x_vals, y_vals, '--', color='r')
    
def xr_proj(x, y, dim):
    x_2 = (x**2).sum(dim)
    cov = (x*y).sum(dim)
    beta = cov/x_2
    return beta

data_dir = '/Users/deanpospisil/Desktop/modules/v4cnn/data/responses/'
da = xr.open_dataset(data_dir + 'taek_tex_shape').load()['resp']

#%%
#pre-processing of data
stim_start = 50
stim_end = 400

stim_dur = (stim_end - stim_start + 50)/1000
time = slice(stim_start, stim_end)
da_t = da.sel(time=time).sum('time', skipna=False)
 
m = da_t.mean('trial', skipna=True).dropna('stim', how='all')
v = da_t.var('trial', skipna=True, ddof=1).dropna('stim', how='all')

means = m.groupby('stim_type').mean('stim')

dyn = np.sqrt(da_t).mean('trial', skipna=True
         ).groupby('stim_type').var('stim', skipna=True, ddof=1)
dt = dyn.sel(stim_type='st')
ds = dyn.sel(stim_type='s')
dif_dyn = dt - ds

coefs = []
exp_nsd0 = []
for var, mean in zip(m, v):
    ind = (var>0)*(mean>0)
    df = pd.DataFrame(np.array([np.log(var.values[ind]), 
                                np.log(mean.values[ind])]).T, 
                                columns=['mean', 'var'])
    mod = (ols('var ~ mean ', data=df).fit())
    c_int = mod.conf_int().loc['mean'].values
    exp_nsd0.append((c_int[0]<1)*(c_int[1]>1))
    df = pd.DataFrame(np.array([var.values[ind], mean.values[ind]]).T, 
                               columns=['mean', 'var'])
    
    modlin = (ols('var ~ mean -1', data=df).fit())
    coefs.append([mod.params[1], np.exp(mod.params[0]), modlin.params[0]])
coefs = np.array(coefs)
dat = np.array([coefs[:,0], coefs[:,1], coefs[:,2], dif_dyn.values, dt.values, 
                ds.values, means.sel(stim_type='st')-means.sel(stim_type='s')]).T
df = pd.DataFrame(dat, columns=['exp', 'slope_exp',  'slope_lin',  'tex_shape', 
                                'tex', 'shape', 'm_ts_dif'])

#%%figures
#%%%
sel_lin = exp_nsd0
df_lin = df[sel_lin]
plt.figure(figsize=(3,3))
plt.scatter(df_lin['tex'], df_lin['shape'],facecolors='none', edgecolors='k')
plt.scatter(df_lin['tex'][64], df_lin['shape'][54], color='g',s=80)
plt.scatter(df_lin['tex'][94], df_lin['shape'][12], color='r', s=80)


plt.xlabel(r'Texture modulation $(Var[\sqrt{t_i}])$', fontsize=12)
plt.ylabel(r'Shape modulation $(Var[\sqrt{s_i}])$', fontsize=12)
plt.axis('square')
plt.xlim(-0.05,1.5);plt.ylim(-0.05,1.5)
plt.plot([-1,1.5], [-1,1.5], color='k')
plt.xticks([0,0.5,1,1.5]);plt.yticks([0,0.5,1,1.5])

plt.tight_layout()

plt.savefig(save_dir+'dyn_ranges.pdf')
#%%
t_ex = 64
s_ex = 27
stim_ind = da_t.coords['stim_type']

plt.figure(figsize=(3.5,2.5))
plt.title('Texture preferring cell')
s1 = da_t[t_ex].mean('trial')[(stim_ind=='s')]
s2 = da_t[t_ex].mean('trial')[(stim_ind=='st')]
plt.plot(s1.coords['stim_ind'].values, s1.values, color='r')
plt.plot(s2.coords['stim_ind'].values+len(s1.values), s2.values, color='g')
plt.legend(['Shapes', 'Textures'])
plt.xlabel('Stimuli index',fontsize=12);plt.ylabel('Response \n (spikes/second)',fontsize=12)
plt.tight_layout()

plt.savefig(save_dir+'tex_cel.pdf')


plt.figure(figsize=(3.5,2.5))
t = 94
plt.title('Shape preferring cell')
stim_ind = da_t.coords['stim_type']
s1 = da_t[s_ex].mean('trial')[(stim_ind=='s')]
s2 = da_t[s_ex].mean('trial')[(stim_ind=='st')]
plt.plot(s1.coords['stim_ind'].values, s1.values,  color='r')
plt.plot(s2.coords['stim_ind'].values+len(s1.values), s2.values, color='g')
#plt.legend(['Shape stimuli', 'Texture stimuli'], fontsize=12)
plt.xlabel('Stimuli index',fontsize=12);
#plt.ylabel('Response (spikes/second)', fontsize=12)
plt.tight_layout()
plt.savefig(save_dir+'shape_cell.pdf')




#%%
#example vmr cells
#sel_lin = (df['exp']<3)&(df['exp']>0)
#plt.figure()
#
#sort_ind = df_lin.index[df_lin['slope_lin'].argsort()]
#rank = 1
#most_var_ind = sort_ind[-rank]
#least_var_ind = sort_ind[rank]
#s=1
#
#most_var_ind = t_ex
#least_var_ind = s_ex
#
#
#
#
#m_i = m[most_var_ind] 
#v_i = v[most_var_ind]
#plt.scatter(m_i, v_i, s=s)
##abline(df['slope_lin'][most_var_ind], 0)
#expline(df['exp'][most_var_ind], df['slope_exp'][most_var_ind])
#
#all_max = np.max([np.max(v_i), np.max(m_i)])
#all_max = all_max*1.1
#plt.xlim([0,all_max]);plt.ylim([0,all_max]);
#plt.plot([0, all_max], [0, all_max])
#plt.axis('square')
#plt.xlabel('Sample Mean (spikes/second)')
#plt.ylabel(r'Sample Variance $(\frac{spikes}{second})^2$')
#
#plt.figure()
#
#m_i = m[least_var_ind] 
#v_i = v[least_var_ind]
#plt.scatter(m_i, v_i, s=s)
##abline(df['slope_lin'][least_var_ind], 0)
#expline(df['exp'][least_var_ind], df['slope_exp'][least_var_ind])
#all_max = np.max([np.max(v_i), np.max(m_i)])
#all_max = all_max*1.1
#plt.xlim([0,all_max]);plt.ylim([0,all_max]);
#plt.plot([0, all_max], [0, all_max])
#plt.xlabel('Sample Mean (spikes/second)')
#plt.ylabel(r'Sample Variance $(\frac{spikes}{second})^2$')
#plt.axis('square')
#

  
#%%
#example vmr cells
def loglog_mv(m, v, exp, slope):
    plt.figure(figsize=(2,2))
    s=1
    no_zero = (m>0)*(v>0)
    m_i = m[no_zero] 
    v_i = v[no_zero]
    plt.scatter(m_i, v_i, s=s)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    #abline(df['slope_lin'][least_var_ind], 0)
    expline(exp, slope)
    all_max = np.max([np.max(v_i), np.max(m_i)])
    all_max = all_max*1.1
    plt.plot([1, all_max], [1, all_max])
    plt.xlim([1, all_max]);plt.ylim([1, all_max]);
    plt.xlabel('Sample Mean (spikes/second)')
    plt.ylabel(r'Sample Variance $(\frac{spikes}{second})^2$')
    #plt.axis('square')

sel_lin = (df['exp']<1.2)&(df['exp']>0.8)
sel_lin = exp_nsd0
df_lin = df[sel_lin]



sort_ind = df_lin.index[df_lin['slope_lin'].argsort()]
rank = 3
most_var_ind = sort_ind[-rank]
least_var_ind = sort_ind[rank]

most_var_ind = t_ex
least_var_ind = s_ex


loglog_mv(m[most_var_ind], v[most_var_ind], 
          1, df['slope_lin'][most_var_ind])
plt.savefig(save_dir+'tex_cell_vmr.pdf')

loglog_mv(m[least_var_ind], v[least_var_ind], 
          1, df['slope_lin'][least_var_ind])
plt.savefig(save_dir+'shape_cell_vmr.pdf')

#%%
#
#loglog_mv(m[least_var_ind], v[least_var_ind], 
#          df['exp'][least_var_ind], df['slope_exp'][least_var_ind])
#
#loglog_mv(m[most_var_ind], v[most_var_ind], 
#          df['exp'][most_var_ind], df['slope_exp'][most_var_ind])
#%%
plt.figure(figsize=(3,2))
plt.scatter(df_lin['slope_lin'], df_lin['tex_shape'], facecolors='none', edgecolors='k')
plt.scatter(df_lin['slope_lin'][s_ex], df_lin['tex_shape'][s_ex], color='r',s=100)
plt.scatter(df_lin['slope_lin'][t_ex], df_lin['tex_shape'][t_ex], color='g', s=100)

plt.xlabel('Variance to mean ratio');plt.ylabel('Texture - shape modulation');
plt.yticks([-1,-.5, 0, .5,1]); plt.xticks([0, .5,1,1.5,2,2.5])
plt.ylim(-1.1,1.1)
mod = (ols('tex_shape ~ slope_lin', data=df_lin).fit())
abline(mod.params.slope_lin, mod.params.Intercept)
plt.title('r='+ str(np.round(mod.rsquared**0.5, 2)) +', F-value=' + str(np.round(mod.fvalue,1))+ ', p=' + str(np.round(mod.f_pvalue,3)))
plt.savefig(save_dir+'tex_shape_vmr.pdf')


print(mod.summary())

 #%%
mod = (ols('m_ts_dif ~ slope_lin', data=df_lin).fit())


