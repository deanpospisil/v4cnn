# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 21:44:16 2016

@author: deanpospisil
"""
import numpy as  np
import scipy.io as  l
import os, sys
#
import matplotlib.pyplot as plt
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')
import xarray as xr
def kurtosis(da):
    da = da.dropna('shapes')
    da = da.transpose('shapes','unit')
    mu = da.mean('shapes')
    sig = da.reduce(np.nanvar, dim='shapes')
    k = (((da - mu)**4).sum('shapes',skipna=True)/da.shapes.shape[0])/(sig**2)
    return k
def measure_TIA_array(unit, m=1,n=1):
    error_space = m*n - m 
    effect_space = m*n - error_space
    s = np.linalg.svd(unit, compute_uv=False)
    numerator = (s[0]**2)/effect_space
    denominator = sum(s[1:]**2)/(error_space)
    return numerator/denominator
def measure_TIA(unit, return_uv=False):
    unit = unit.dropna('x', 'all').dropna('shapes', 'all')
    unit = unit.transpose('shapes','x')
    if return_uv:
        u,s,v = np.linalg.svd(unit.values, compute_uv=True)
        return (s[0]**2)/np.sum(s**2), u , s,  v
    else:
        s = np.linalg.svd(unit.values, compute_uv=False)
        return (s[0]**2)/np.sum(s**2)
    
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
fn = top_dir +'data/responses/v4_ti_resp.nc'
v4 = xr.open_dataset(fn)['resp'].load()
v4 = v4.transpose('unit', 'x', 'shapes') 
v4 = v4 - v4.mean('shapes')
av_cov = np.array([norm_av_cov(unit) for unit in v4])
tia = np.array([measure_TIA(unit) for unit in v4])
k = kurtosis(v4.var('x'))

for ti_in in np.argsort(av_cov)[-5::1]:
    plt.figure(figsize=(10,2))
    v4_resp =  v4[ti_in].dropna('x', 'all').dropna('shapes', 'all')
    plt.imshow(v4_resp, interpolation='nearest', cmap=plt.cm.Greys_r)
    plt.colorbar()
    plt.title('av_cov = ' + str(np.round(av_cov[ti_in], 3)) + 
              'TIA: ' + str(np.round(tia[ti_in], 2))) 

    plt.xlabel('shape');plt.ylabel('pos')
    
    _, u,s, v = measure_TIA(v4[ti_in], return_uv=True)
    r_hat = s[0]*np.dot(u[:,0].reshape(len(u),1), v[0,:].reshape(1,len(v))).T
    plt.figure(figsize=(10,2))
    plt.imshow(r_hat, interpolation='nearest', cmap=plt.cm.Greys_r)
    plt.colorbar()
    plt.title('R estimate');plt.xlabel('shape');plt.ylabel('pos')
    
    plt.figure(figsize=(10,2))
    plt.imshow(v4_resp - r_hat, interpolation='nearest', cmap=plt.cm.Greys_r)
    plt.colorbar();plt.title('R-R_hat');plt.xlabel('shape');plt.ylabel('pos')
    
    plt.figure(figsize=(6,2))
    plt.subplot(121);plt.plot(u[:,0]);plt.xlabel('shape'); plt.ylabel('SP')
    plt.subplot(122);plt.plot(v[0,:]*s[0]);plt.xlabel('pos');
    plt.ylabel('RF');plt.tight_layout()
    
    plt.figure()
    plt.imshow(np.corrcoef(v4_resp),interpolation='nearest', 
               cmap=plt.cm.Greys_r)
    plt.colorbar()

'''
n_s = np.arange(5,100)
m=30
iterations = range(100)
for pc in [0,1,2]:
    F_mean = [];
    for n in n_s:
        Fs=[]
        for ind in iterations:
            mat = np.random.normal(size=(m,n))
            mat -= np.mean(mat, 0)
    
            constant = 0
            df_reg = n
            error_space = m*n - df_reg - constant
            effect_space = m*n - error_space       
            
            if pc==1:            
                i_variable = np.random.normal(size=(m,1))
            elif pc==0:
                i_variable = np.linalg.svd(mat, compute_uv=True)[0][:,0].reshape(30,1)
            else:
                i_variable = np.random.normal(size=(m,1))
                error_space = 1
                effect_space = 1 
            i_variable -= np.mean(i_variable)
            
            x = np.linalg.lstsq(i_variable, mat)[0]
            b_hat = np.dot(i_variable, x)
            m_y = np.sum(b_hat**2) / effect_space
            m_e = np.sum((b_hat - mat)**2) / error_space
            Fs.append(m_y / m_e)
        F_mean.append(np.mean(Fs))
    
    plt.plot(n_s, F_mean)
    
plt.legend(['PC','IID normal', 'PC unscaled'], title='Independent Variable')
plt.xlabel('n rows')
plt.ylabel('F-value')

n_s = np.arange(5,20)
m=30
iterations = range(100)
for pc in [0,1]:
    F_mean = [];
    for n in n_s:
        Fs=[]
        print(n)
        for ind in iterations:
            mat = np.random.normal(size=(m,n))
            mat -= np.mean(mat, 0)
    
            constant = 0
            df_reg = n
            error_space = m*n - df_reg - constant
            effect_space = m*n - error_space       
            
            if pc:            
                i_variable = np.random.normal(size=(m,1))
            else:
                i_variable = np.linalg.svd(mat, compute_uv=True)[0][:,0].reshape(30,1)
            i_variable -= np.mean(i_variable)
            
            x = np.linalg.lstsq(i_variable, mat)[0]
            b_hat = np.dot(i_variable, x)
            m_y = np.sum(b_hat**2) / effect_space
            m_e = np.sum((b_hat - mat)**2) / error_space
            Fs.append(m_y / m_e)
        F_mean.append(np.mean(Fs))
    
    plt.plot(n_s, F_mean)
plt.legend(['PC','IID normal'], title='Dependent Variable')
plt.xlabel('n rows')
plt.ylabel('F-value')


n_s = np.arange(3,20)
m=30
iterations = range(2000)
tins_s = [];
for n in n_s:
    temp=[]
    print(n)
    for ind in iterations:
        mat = np.random.normal(size=(m,n))
        mat -= np.mean(mat, 0)
        
        constant = 0
        df_reg = n
        error_space = m*n - df_reg - constant
        effect_space = m*n - error_space       
        
        i_variable = np.random.normal(size=(m,1))
        i_variable -= np.mean(i_variable)
        
        x = np.linalg.lstsq(i_variable, mat)[0]
        b_hat = np.dot(i_variable, x)
        m_y = np.sum(b_hat**2)/effect_space
        m_e = np.sum((b_hat-mat)**2)/error_space
        temp.append(m_y/m_e)
    tins_s.append(np.mean(temp))

plt.plot(n_s, tins_s)

m=25.
iterations = range(5000)
temp = []
e_s = []
b_hats_s = []
for ind in iterations:
    d_variable = np.random.normal(size=(m,1))
    d_variable -= np.mean(d_variable)
    
    constant = 0
    df_reg = 1
    error_space = m - df_reg - constant
    effect_space = m - error_space       
    
    i_variable = np.random.normal(size=(m,1))
    i_variable -= np.mean(i_variable)
    #i_variable = np.concatenate((i_variable, np.ones((m,1))),1)
    
    x = np.linalg.lstsq(i_variable, d_variable)[0]
    b_hat = np.dot(i_variable, x)
    m_y = np.sum(b_hat**2)/effect_space
    m_e = np.sum((b_hat-d_variable)**2)/error_space
    temp.append(m_y/m_e)
    b_hats_s.append(np.sum(b_hat**2))
    e_s.append(np.sum((b_hat-d_variable)**2))

plt.plot(iterations, temp)
print(np.mean(temp))
print(np.mean(e_s))
print(np.mean(b_hats_s))


'''
