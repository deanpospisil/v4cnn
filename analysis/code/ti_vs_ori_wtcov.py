#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:57:49 2017

@author: dean
"""

import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn')
sys.path.insert(0, top_dir + 'xarray/');
top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr
import pandas as pd
cnn_name = 'bvlc_reference_caffenet_sinusoids'
data_dir = '/loc6tb/'


imshape = (227, 227)
rft = np.fft.rfft2(np.ones(imshape))
rft_shape = rft.shape

r = np.fft.fftfreq(imshape[0]).reshape((imshape[0]), 1)
c = np.fft.fftfreq(imshape[1]).reshape((1, imshape[1]))

cart_c = c + r*1j
freq_index = cart_c[:rft_shape[0], :rft_shape[1]].ravel()

#%%
import d_net_analysis as na
y_nm = 'bvlc_reference_caffenetpix_width[32.0]_x_(114.0, 114.0, 1)_y_(64, 164, 51)_amp_NonePC370.nc'
x_nm = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370.nc'
xy_nm = 'bvlc_reference_caffenetpix_width[32.0]_x_(64.0, 164.0, 21)_y_(64.0, 164.0, 21)_amp_NonePC370.nc'
ti = []
k = []
for net_name in [y_nm, x_nm, xy_nm]:
    da = xr.open_dataset(data_dir + '/data/responses/'+ net_name)['resp'].squeeze()
    ind = da.coords['layer_label']=='conv2'
    #k.append(na.kurtosis_da(da[..., ind]))
    ti.append(na.ti_in_rf(da[..., ind]))
#%%  
import pickle
goforit = True      
if 'a' not in locals() or goforit:
    with open(top_dir + 'nets/netwts.p', 'rb') as f:    
        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)
            
wts = a[0][1]

#%%
wts = a[1][1]
ti_xy = ti[2]
ti_x = ti[1]
ti_y = ti[0]
ti_x = ti_x[ti_x.coords['layer_label']=='conv2']
ti_y = ti_y[ti_y.coords['layer_label']=='conv2']
ti_xy = ti_xy[ti_xy.coords['layer_label']=='conv2']

plt.subplot(211)
plt.scatter(ti_x, ti_y)
plt.axis('square')

plt.subplot(212)
plt.scatter(ti_x, ti_xy)
plt.axis('square')
#%%
da = xr.open_dataset(data_dir + '/data/responses/'+ net_name)['resp'].squeeze()
plt.figure(figsize=(2,8))
dims = ['unit','chan', 'y', 'x']
layers = range(5)[1:]
#layers = [1,]
for ind in layers:
    netwtsd = {}
    layer_labels = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6']
    layer = a[ind][1]
    dim_names = dims[:len(layer.shape)]
    layer_ind = da.coords['layer_label'].values == layer_labels[ind] 
    the_coords =  da[..., layer_ind].coords['unit']
    layer_da = xr.DataArray(layer, dims=dims, 
           coords=[range(n) for n in np.shape(layer)])
    kw = layer.shape[-1]
    k_mid = int(kw/2.)
    wt_cov_y = []
    for row in range(layer.shape[-1]):
        layer_da = layer_da.transpose('unit', 'chan', 'y', 'x')
        data = layer_da.values[..., row, :]
        cov = np.matmul(data.transpose(0, 2, 1), data)
        cov = cov.sum(axis=(1,2)) - np.trace(cov, axis1=1, axis2=2)
        
        
        vnorm = np.linalg.norm(data, axis=1)
        outer_prod = (vnorm[:, :, np.newaxis])*(vnorm[:, np.newaxis, :])
        
        outer_prod = outer_prod.sum(axis=(1,2)) - np.trace(outer_prod, axis1=1, axis2=2)
        opponency = cov / outer_prod
        
        opponency_da = xr.DataArray(opponency, dims=('unit',))
        opponency_da.coords['unit'] = layer_da.coords['unit']
        wt_cov_y.append(opponency_da)
    
    wt_cov_x = []
    for row in range(layer.shape[-1]):
        layer_da = layer_da.transpose('unit', 'chan', 'y', 'x')
        data = layer_da.values[..., :, row]
        cov = np.matmul(data.transpose(0, 2, 1), data)
        cov = cov.sum(axis=(1,2)) - np.trace(cov, axis1=1, axis2=2)
        
        
        vnorm = np.linalg.norm(data, axis=1)
        outer_prod = (vnorm[:, :, np.newaxis])*(vnorm[:, np.newaxis, :])
        
        outer_prod = outer_prod.sum(axis=(1,2)) - np.trace(outer_prod, axis1=1, axis2=2)
        opponency = cov / outer_prod
        
        opponency_da = xr.DataArray(opponency, dims=('unit',))
        opponency_da.coords['unit'] = layer_da.coords['unit']
        wt_cov_x.append(opponency_da)
    plt.subplot(4,1,ind)
    plt.scatter(wt_cov_y[k_mid].values, wt_cov_x[k_mid].values, color='k', marker='o', s=0.7)
    plt.axis('square');plt.xlim(-.3,1);plt.ylim(-.3,1);
    plt.plot([0,1], [0,1])
    plt.title(layer_labels[ind])
plt.xlabel('Wt. Cov. X '); plt.ylabel('Wt. Cov. Y')

plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/ori_ti/wt_cov_anis_goes_down_w_layer.pdf')

#%%
def spatial_opponency(da):
    da = da.transpose('unit', 'chan', 'y', 'x')
    data = da.values.reshape(da.shape[:2] + (np.product(da.shape[2:]),))
    cov = np.matmul(data.transpose(0, 2, 1), data)
    cov = cov.sum(axis=(1,2)) - np.trace(cov, axis1=1, axis2=2)

    
    vnorm = np.linalg.norm(data, axis=1)
    outer_prod = (vnorm[:, :, np.newaxis])*(vnorm[:, np.newaxis, :])

    outer_prod = outer_prod.sum(axis=(1,2)) - np.trace(outer_prod, axis1=1, axis2=2)
    opponency = cov / outer_prod
    
    opponency_da = xr.DataArray(opponency, dims=('unit',))
    opponency_da.coords['unit'] = da.coords['unit']
    
    return opponency_da

def norm_cov(x, subtract_mean=True):
    
    #if nxm the get cov nxn
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 0, keepdims=True)
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    vnrm = np.linalg.norm(x, axis=0, keepdims=True)
    denominator = np.sum(np.multiply(vnrm.T, vnrm)[diag_inds]) 
    norm_cov = numerator/denominator
    
    return norm_cov

ti_xysum = ti_x + ti_y
ti_xysum_ind = ti_xysum>0

wt_cov_x = []
wt_cov_y = [] 
wt_cov_xy = []
for kern in wts:
    kern = kern.T
    ncx=[]
    ncy=[]
    for row in kern:
        ncx.append(norm_cov(row))
    for col in np.swapaxes(kern, 0, 1):
        ncy.append(norm_cov(col))
    wt_cov_x.append(ncx)
    wt_cov_y.append(ncy)
    wt_cov_xy.append(norm_cov(kern.reshape(25, 48)))

#%% 
wt_cov_x = np.array(wt_cov_x).T
wt_cov_y = np.array(wt_cov_y).T
#%%
row = 3
plt.figure(figsize=(3,12))
plt.subplot(511)
plt.scatter(wt_cov_xy, ti_xy)
plt.xlabel('wc all')

plt.subplot(512)
plt.scatter(wt_cov_y[:,row], ti_x)
plt.xlabel('wc x')

plt.subplot(513)
plt.scatter(wt_cov_y[:,row], ti_y)
plt.xlabel('wc x')

plt.subplot(514)
plt.scatter(wt_cov_x[:,row], ti_x)
plt.xlabel('wc y')

plt.subplot(515)
plt.scatter(wt_cov_x[:,row], ti_y)
plt.xlabel('wc y')
#%%
data = np.concatenate([np.array(wt_cov_xy)[:, np.newaxis], wt_cov_x[:,2:3],
                       wt_cov_y[:, 2:3], ti_xy.values[:, np.newaxis], 
                       ti_x.values[:, np.newaxis], ti_y.values[:,np.newaxis]],axis=1)
data=data.T
ti_xysum = ti_x + ti_y
ti_xysum_ind = ti_xysum>1

ti_x_better_ind = (ti_x>np.percentile(ti_x,80))*(ti_y<np.percentile(ti_y,20))

ti_y_better_ind = (ti_y>np.percentile(ti_y,80))*(ti_x<np.percentile(ti_x,50))
other = (ti_xysum_ind.values + ti_x_better_ind.values + ti_y_better_ind.values)==0

highlight_inds = [other,
                  ti_xysum_ind.values, ti_x_better_ind.values, 
                  ti_y_better_ind.values, 
                  ]
names = ['wxy', 'wy', 'wx', 'txy', 'tx', 'ty']
import itertools
import numpy as np
import matplotlib.pyplot as plt
def scatterplot_matrix(data, names, highlight_inds, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
    for highlight_ind in highlight_inds:
        # Plot the data.
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i,j), (j,i)]:
                axes[x,y].plot(data[x, highlight_ind],
                            data[y, highlight_ind], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig
scatterplot_matrix(data,names,highlight_inds,linestyle='none', markersize=4, marker='o', mfc='none')
plt.savefig(top_dir + 'analysis/figures/images/ori_ti/wt_cov_explains_diff_ori_no_other')
#%%
wt_cov_x = wt_cov_x[2]
wt_cov_y = wt_cov_y[2]
#%%
ti_xysum = ti_x + ti_y
ti_xysum_ind = ti_xysum>1

ti_x_better_ind = (ti_x>np.percentile(ti_x,75))*(ti_y<np.percentile(ti_y,30))

ti_y_better_ind = (ti_y>np.percentile(ti_y,80))*(ti_x<np.percentile(ti_x,20))
other = (ti_xysum_ind.values + ti_x_better_ind.values + ti_y_better_ind.values)==0

highlight_inds = [other,
                  ti_xysum_ind.values, ti_x_better_ind.values, 
                  ti_y_better_ind.values, 
                  ]
marker = 'o';markersize=3
colors = ['b', 'r', 'c', 'y']
for color, highlight_ind in zip(colors, highlight_inds):
    plt.subplot(212)
    plt.scatter(wt_cov_x[highlight_ind], wt_cov_y[highlight_ind], 
                color=color, marker=marker, s=markersize)
    plt.axis('square');plt.xlim(-.3,1);plt.ylim(-.3,1);
    plt.xlabel('Wt. Cov. X '); plt.ylabel('Wt. Cov. Y')
    plt.subplot(211)
    plt.title('Conv2 Anisotropy')
    plt.scatter(ti_y[highlight_ind], ti_x[highlight_ind], 
                color=color, marker=marker, s=markersize)
    plt.axis('square');plt.xlim(-.1,1);plt.ylim(-.1,1);
    plt.xlabel('TI X '); plt.ylabel('TI Y')
    plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/ori_ti/wt_cov_explains_anisotropy.pdf')
#%%
#%%

#%%
def spatial_opponency(da):
    da = da.transpose('unit', 'chan', 'y', 'x')
    data = da.values.reshape(da.shape[:2] + (np.product(da.shape[2:]),))
    cov = np.matmul(data.transpose(0, 2, 1), data)
    cov = cov.sum(axis=(1,2)) - np.trace(cov, axis1=1, axis2=2)

    
    vnorm = np.linalg.norm(data, axis=1)
    outer_prod = (vnorm[:, :, np.newaxis])*(vnorm[:, np.newaxis, :])

    outer_prod = outer_prod.sum(axis=(1,2)) - np.trace(outer_prod, axis1=1, axis2=2)
    opponency = cov / outer_prod
    
    opponency_da = xr.DataArray(opponency, dims=('unit',))
    opponency_da.coords['unit'] = da.coords['unit']
    
    return opponency_da
import pandas as pd
wt_cov_by_layer = []
for layer, layer_name in zip(netwtsd, layer_labels):
    print(layer[1].shape)
    if len(layer[1].shape)>2:
        _ = xr.DataArray(layer[1], dims=['unit', 'chan', 'x', 'y'])
        wt_cov = spatial_opponency(_)
        print(len(wt_cov))
        wt_cov_by_layer.append(wt_cov)
wt_covs = np.concatenate(wt_cov_by_layer)

non_k_var = (kurt_shapes_yx<42) * (kurt_shapes_yx>2) * (tot_vars_yx>0) 

keys = ['layer_label', 'unit']
coord = [da.coords[key].values for key in keys]
index = pd.MultiIndex.from_arrays(coord, names=keys)
ti = pd.DataFrame(np.hstack([ti_yx,]), index=index, columns=['ti',])    