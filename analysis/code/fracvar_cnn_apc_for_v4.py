#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:51:10 2017

@author: dean
"""
import sys, os
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn')
sys.path.insert(0, top_dir + 'xarray/');
import xarray as xr 
import numpy as np

top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common')
import apc_model_fit as ac
import matplotlib.pyplot as plt
#%%
data_dir = '/loc6tb/'
def cor2(a,b):
    if len(a.shape)<=1:
        a = np.expand_dims(a,1)
    if len(b.shape)<=1:
        b = np.expand_dims(b,1)
    a -= a.mean(0);
    b -= b.mean(0)
    a /= np.linalg.norm(a, axis=0);
    b /= np.linalg.norm(b, axis=0);
    corrcoef = np.dot(a.T, b)       
    return corrcoef
 
#v4 fit to CNN and APC
v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(data_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
file = open(data_dir + 'data/responses/v4_apc_109_neural_labels.txt', 'r')
wyeth_labels = [label.split(' ')[-1] for label in 
            file.read().split('\n') if len(label)>0]
v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
fn = data_dir + 'data/models/' + 'apc_models_362.nc'

if 'apc_fit_v4' not in locals():
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()
    apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                      dmod.chunk({}), 
                                      fit_over_dims=None, 
                                      prov_commit=False)**2

cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
            'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)']

cnn_models = []
for cnn_name in cnn_names:
    da_temp = xr.open_dataset(data_dir + 'data/responses/' + cnn_name + '.nc')['resp']
    da_temp = da_temp.sel(unit=slice(0, None, 1)).squeeze()
    middle = 114
    da_0_temp = da_temp.sel(x=middle)
    da_0_temp = da_0_temp.sel(shapes=v4_resp_apc.coords['shapes'].values)
    cnn_models.append(da_0_temp)


#direct fit
model_ind_lists = []
cor_v4_models = []
cnn_fit_v4_models = []
cnn_fit_v4_model_inds = []
for model in cnn_models:
    cnn_fit_v4 = cor2(model.values, v4_resp_apc.values)
    cnn_fit_v4[np.isnan(cnn_fit_v4)] = 0
    cnn_fit_v4_model_ind = cnn_fit_v4.argmax(0)
    model_cor = cnn_fit_v4.max(0)
    
    cnn_fit_v4_models.append(model_cor)
    cnn_fit_v4_model_inds.append(cnn_fit_v4_model_ind)

cnn_fit_apc_models = []
cnn_fit_apc_model_inds = []
for model in cnn_models:
    cnn_fit_apc = cor2(model.values, dmod[:, apc_fit_v4.models.values].values)
    cnn_fit_apc[np.isnan(cnn_fit_apc)] = 0
    cnn_fit_apc_model_ind = cnn_fit_apc.argmax(0)
    model_cor = cnn_fit_apc.max(0)
    
    cnn_fit_apc_models.append(model_cor)
    cnn_fit_apc_model_inds.append(cnn_fit_apc_model_ind)

#%%
fig_dir = '/home/dean/Desktop/v4cnn/analysis/figures/images/frac_var_cnn_apc/'

apc_bf = dmod[:, apc_fit_v4.models.values]
cnn_train_bf = cnn_models[0][:, cnn_fit_apc_model_inds[0]]
cnn_untrain_bf = cnn_models[0][:, cnn_fit_apc_model_inds[1]]


apc_bf = apc_bf - apc_bf.mean('shapes')
apc_bf = apc_bf/((apc_bf**2).sum('shapes')**0.5)


cnn_train_bf = cnn_train_bf - cnn_train_bf.mean('shapes')
cnn_train_bf = cnn_train_bf/((cnn_train_bf**2).sum('shapes')**0.5)
cnn_r2 = []
apc_orth_cnn_r2 = []
apc_r2 = []
for unit in range(109):
    
    apc = apc_bf[:,unit].values
    cnn = cnn_train_bf[:,unit].values
    v4 = v4_resp_apc[:, unit].values
    #x, res, rank, s = np.linalg.lstsq(np.expand_dims(cnn, 1), 
    #                                  np.expand_dims(apc, 1))
    
    x = np.dot(apc, cnn)
    apc_perp_cnn = apc - cnn*x
    apc_perp_cnn = apc_perp_cnn/(np.sum(apc_perp_cnn**2)**0.5)
    
    
    a = np.dot(v4, cnn)
    b = np.dot(v4, apc_perp_cnn)
    abproj = a*cnn + b*apc_perp_cnn
    
    error = abproj - v4
    error_len = np.sum(error**2)
    a_len = a
    b_len = b
    cnn_r2.append(a_len)
    apc_orth_cnn_r2.append(b_len)
    apc_r2.append(np.dot(apc, v4))



plt.figure()
plt.scatter(apc_orth_cnn_r2, apc_r2)
plt.xlabel('APC ORTH CNN (R)')
plt.ylabel('APC (R2)')
plt.ylim(0,1);plt.xlim(0,1)
plt.plot([0,1], [0,1])
plt.axis('square')
plt.savefig(fig_dir + 'APC ORTH CNN and APC.pdf')


#%%
cnn_r2 = []
cnn_orth_apc_r2 = []
apc_r2 = []
for unit in range(109):
    
    apc = apc_bf[:,unit].values
    cnn = cnn_train_bf[:,unit].values
    v4 = v4_resp_apc[:, unit].values

    
    x = np.dot(apc, cnn)
    cnn_perp_apc = cnn - apc*x
    cnn_perp_apc = cnn_perp_apc/(np.sum(cnn_perp_apc**2)**0.5)
    cnn_pred = np.array([cnn, cnn_perp_apc]).T
    x, res, rank, s = np.linalg.lstsq(cnn_pred, 
                                     np.expand_dims(apc, 1))
    a = x[0]
    b = x[1]
    abproj = a*cnn + b*cnn_perp_apc
    
    error = abproj - v4
    error_len = np.sum(error**2)
    a_len = a**2
    b_len = b**2
    cnn_r2.append(np.dot(cnn, v4))
    cnn_orth_apc_r2.append(np.abs(np.dot(cnn_perp_apc, v4)))
    apc_r2.append(np.dot(apc, v4))

plt.figure()
plt.scatter(np.array(cnn_r2), np.array(apc_r2))
plt.xlabel('CNN(R)')
plt.ylabel('APC (R)' )
plt.ylim(0,1);plt.xlim(0,1)
plt.plot([0,1], [0,1])
plt.axis('square')
plt.savefig(fig_dir + 'CNN and APC.pdf')


plt.figure()
plt.scatter(cnn_orth_apc_r2, apc_r2)
plt.xlabel('CNN ORTH APC (R)')
plt.ylabel('APC (R)')# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
plt.ylim(0,1);plt.xlim(0,1)
plt.plot([0,1], [0,1])
plt.axis('square')
plt.savefig(fig_dir + 'CNN ORTH APC and APC.pdf')

#%%
red_below = [16,51,52,55,63,64,65,67,77,101]
apc_bf = dmod[:, apc_fit_v4.models.values]
cnn_train_bf = cnn_models[0][:, cnn_fit_v4_model_inds[0]]
cnn_untrain_bf = cnn_models[0][:, cnn_fit_v4_model_inds[1]]


apc_bf = apc_bf - apc_bf.mean('shapes')
apc_bf = apc_bf/((apc_bf**2).sum('shapes')**0.5)


cnn_train_bf = cnn_train_bf - cnn_train_bf.mean('shapes')
cnn_train_bf = cnn_train_bf/((cnn_train_bf**2).sum('shapes')**0.5)
cnn_r2 = []
cnn_orth_apc_r2 = []
apc_r2 = []
for unit in range(109):
    
    apc = apc_bf[:,unit].values
    cnn = cnn_train_bf[:,unit].values
    v4 = v4_resp_apc[:, unit].values

    
    x = np.dot(apc, cnn)
    cnn_perp_apc = cnn - apc*x
    cnn_perp_apc = cnn_perp_apc/(np.sum(cnn_perp_apc**2)**0.5)
    cnn_pred = np.array([cnn, cnn_perp_apc]).T
    x, res, rank, s = np.linalg.lstsq(cnn_pred, 
                                     np.expand_dims(apc, 1))
    a = x[0]
    b = x[1]
    abproj = a*cnn + b*cnn_perp_apc
    
    error = abproj - v4
    error_len = np.sum(error**2)
    a_len = a**2
    b_len = b**2
    cnn_r2.append(np.dot(cnn, v4))
    cnn_orth_apc_r2.append(np.abs(np.dot(cnn_perp_apc, v4)))
    apc_r2.append(np.dot(apc, v4))




plt.figure()
plt.scatter(cnn_orth_apc_r2, apc_r2)
plt.scatter(np.array(cnn_orth_apc_r2)[np.array(red_below)], np.array(apc_r2)[np.array(red_below)], color='red')

plt.xlabel('CNN ORTH APC (R)')
plt.ylabel('APC (R)')
plt.ylim(0,1);plt.xlim(0,1)
plt.plot([0,1], [0,1])
plt.axis('square')
plt.savefig(fig_dir + 'CNN ORTH APC and APC_for_cnn_not_dirfit_apc.pdf')

print(np.median(cnn_orth_apc_r2))
print(np.median(apc_r2))
print(np.median(np.array(apc_r2)-cnn_orth_apc_r2))
print(sum((np.array(apc_r2)-cnn_orth_apc_r2)>0))



#%%
plt.scatter(apc_orth_cnn_r2, apc_r2);
plt.xlabel('APC ORTH CNN (R)')
plt.ylabel('APC (R)')
plt.ylim(0,1);plt.xlim(0,1)
plt.plot([0,1], [0,1])
plt.axis('square')
