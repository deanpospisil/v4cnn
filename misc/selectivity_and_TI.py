# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:36:19 2017

@author: deanpospisil
"""

import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
import pickle
layer_labels_b = [b'conv2', b'conv3', b'conv4', b'conv5', b'fc6']
layer_labels = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6']


top_dir = os.getcwd().split('v4cnn')[0]
top_dir = top_dir + 'v4cnn'

with open(top_dir + '/data/an_results/ti_vs_wt_cov_exps_.p', 'rb') as f:    
    try:
        an = pickle.load(f, encoding='latin1')
    except:
        an = pickle.load(f)
        
with open(top_dir + '/nets/netwtsd.p', 'rb') as f:    
    try:
        netwtsd = pickle.load(f, encoding='latin1')
    except:
        netwtsd = pickle.load(f)
        
ti = an[0][0] 
wt_cov = an[1][0] 
layer_labels_ind = np.array(list(map(str, ti.coords['layer_label'].values)))



        
#positivity
positivity = []
for layer in layer_labels:
    a_layer = netwtsd[layer]
    positivity.append(a_layer.sum(['chan', 'y', 'x'])/np.abs(a_layer).sum(['chan', 'y', 'x']))
#%%
for i, layer in enumerate(layer_labels_b):
    plt.figure()
    x = positivity[i]
    y = ti[ti.coords['layer_label'].values==layer]
    x, y = xr.align(x, y, join='inner')
    plt.title(np.round(np.corrcoef(x,y)[0,1],2))
    plt.scatter(x, y, s=1)
    plt.xlim(-1,1)
    plt.ylim(0,1)
    
#%%
for i, layerb, layer  in zip(range(len(positivity)), layer_labels_b, layer_labels):
    plt.figure()
    reg1 = positivity[i]
    reg2 = wt_cov[wt_cov.coords['layer_label'].values==layer]
    x, y = xr.align(reg1, reg2, join='inner')
    regs = np.vstack([x,y, np.ones(np.shape(x))])
    regs = np.vstack([x, np.ones(np.shape(x))])

    
    y = ti[ti.coords['layer_label'].values==layerb].values
    x ,res, ran, s = np.linalg.lstsq(regs.T, y)
    pred = np.dot(regs.T, x)
    plt.title(np.round(np.corrcoef(pred,y)[0,1],2))
    plt.scatter(pred, y, s=1)
    plt.xlim(-1,1)
    plt.ylim(0,1)

