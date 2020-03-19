import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#%%
data_dir = '/Users/deanpospisil/Desktop/modules/v4cnn/data/responses/plex_resp/'
df = xr.open_dataset(data_dir + 'l180912_deans_task_01_01_recut.nc')
di = df['im'].load()
da = df['resp'].load()

#%%
pre_stim= 0.15
post_stim = 0.35
count_time = post_stim - pre_stim
plt.figure(figsize=(10,10))
da.mean(['trial']).plot()
da_s = da[:,:10,:].loc[..., pre_stim:post_stim].sum('t')

dasm = da_s.mean('trial');
#%%
da_s = da[:,:10,:].loc[..., pre_stim:post_stim].sum('t')

good_trials = list(range(2)) + list(range(4,10))
good_trials = range(10)
da_s = da_s[:,good_trials]
m, v = da_s.mean('trial'), da_s.var('trial')
plt.scatter(m, v, s=1)
plt.title(np.round((v/m).mean().values, 2))
#plt.axis('equal')
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
plt.axis('square')
plt.xlim([1,1000]);plt.ylim([1,1000])
plt.plot([0,1000], [0,1000], c='r');
plt.ylabel('Variance');plt.xlabel('mean')
#%%
t = np.linspace(-0.1, .5, 300)
d = da.groupby_bins('t', bins=t, labels=t[1:]).sum('t')
d.mean(('trial', 'stim'))[:150].plot()
#%%
sort_inds = dasm.argsort().values[::-1]
sort_ims = di[sort_inds].values
resp = dasm[sort_inds].values
inds = -np.arange(60)
plt.figure(figsize=(8,60))
for im, a_resp, imnum, a, i in zip(sort_ims[inds], resp[inds], 
                             sort_inds[inds], 
                             di[...,-1].max(['r','c'])[sort_inds][inds].values
                             , inds):
    plt.subplot(len(inds),1,-i+1)
    plt.xticks([]);plt.yticks([])
    plt.imshow(im[10:-10,10:-10]/255)
    plt.ylabel(str(int(a_resp)), rotation=0 )
    #plt.xlabel(imnum)
    #plt.ylabel(a)
plt.tight_layout()
#%%
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
#%%
i=[]
s=[]
b = []
post_stims = np.linspace(0.05, 0.5, 20)
for post_stim in post_stims:
    da_s = da[:, :10, :].loc[..., pre_stim:post_stim].sum('t')
    
    m = np.log10(da_s.mean('trial'))
    v = np.log10(da_s.var('trial'))
    
    
    dat = pd.DataFrame(np.array([m,v]).T, columns=['m','v'])
    results = smf.ols('v ~ m', data=dat).fit()
    i.append(results.params[0])
    s.append(results.params[1])
    #plt.figure()
    m = 10.**m
    v = 10.**v
    r = (v/m).mean('stim')
    b.append(600*np.var((m**(1/(2*r)).values)))

    #plt.scatter(m,v)
    #plt.plot(np.arange(1,np.max(v)), 
             #(10.**results.params[0])*(np.arange(1,np.max(v))**results.params[1]),
             #c='r')
    
    #ax = plt.gca()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #plt.axis('square')
    #plt.xlim([1,np.max(v)]);plt.ylim([1,np.max(v)])
    #plt.plot([1,np.max(v)], [1,np.max(v)], c='r');
#%%
i = np.array(i)
s = np.array(s)
plt.subplot(211)
plt.plot(post_stims, 10**i)
plt.plot(post_stims, s)

plt.legend(['slope', 'exponent'])
plt.subplot(212)
plt.plot(post_stims, b)
#%%
bins=np.linspace(0.0,.5, 40)
db = da[:,:10,:].groupby_bins('t', bins=bins).sum('t')
m = db.mean('trial')
v = db.var('trial')
plt.subplot(111)
r = (v/m).mean('stim')
plt.plot(bins[:-1],r)
plt.xlabel('time (sec)', fontsize=20)
plt.ylabel(r'$\frac{\sigma^2}{\mu}$', rotation=0, fontsize=30, labelpad=20)
plt.plot([0,0.3], [0.7,0.7])
plt.legend([r'$\frac{\sigma^2}{\mu}$', 'stim on'], fontsize=16)
plt.tight_layout()
plt.savefig('/Users/deanpospisil/Desktop/writing/science/v4cnn_paper_elife/rebuttal/mv_v4_time.pdf')
#plt.subplot(212)
#bins=np.linspace(0.0,.5, 30)
#db = da[:,:10,:].groupby_bins('t', bins=bins).sum('t')
#
#plt.plot(bins[:-1], db.mean(['trial', 'stim'])* (30))
#plt.tight_layout()
#%%
(m,v) = (m[:, 2], v[:, 2])
#%%
plt.figure()
ax = plt.gca()
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.axis('square')
plt.xlim([1,np.max(v)]);plt.ylim([1,np.max(v)])
plt.plot([1,np.max(v)], [1,np.max(v)], c='k');
plt.scatter(m, v)

dat = pd.DataFrame(np.array([m,v]).T, columns=['m','v'])
results = smf.ols('v ~ m', data=np.log10(dat)).fit()
plt.plot(np.arange(1,np.max(v)), 
         (10.**results.params[0])*(np.arange(1,np.max(v))**results.params[1]),
         c='r')
#%%
for i in range(40):
    plt.figure()
    plt.plot(bins[:-1],(np.corrcoef(m.T)**2)[i,:]);
