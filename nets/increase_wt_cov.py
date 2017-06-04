#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:46:11 2017

@author: dean
"""
import numpy as np
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir + 'v4cnn'
import xarray as xr
import pandas as pd

def prin_comp_maps(da):
    da = da.transpose('unit', 'chan', 'y', 'x')

    data = da.values.reshape(da.shape[:2] + (np.product(da.shape[2:]),))
    u, s, v = np.linalg.svd(data, full_matrices=False)
    v = v.reshape(v.shape[:2]+ da.shape[-2:])#reshape into space
    
    u_da = xr.DataArray(u, dims=('unit', 'chan', 'pc'), 
                        coords=[range(n) for n in np.shape(u)])
    u_da.coords['unit'] = da.coords['unit']
    s_da = xr.DataArray(s, dims=('unit', 'sv'), 
                        coords=[range(n) for n in np.shape(s)])
    s_da.coords['unit'] = da.coords['unit']
    v_da = xr.DataArray(v, dims=('unit', 'chan', 'x', 'y'), 
                        coords=[range(n) for n in np.shape(v)])
    v_da.coords['unit'] = da.coords['unit']
    
    return u_da, s_da, v_da


def adjust_to_r(static_vector, shift_vector, desired_r):
    #center both vectors
    if np.corrcoef(shift_vector, static_vector)[0,1]<0:
        shift_vector = -shift_vector
    b = shift_vector
    b = b - b.mean()
    x = static_vector
    x = x - x.mean()
    
    #project x onto b
    s = np.dot(x, b)/np.linalg.norm(b)**2
    b = b * s
    #get the error vector, or vecto pointing from tip of b to x
    e = x - b
    #convert desired r to angle    
    theta = np.arccos(desired_r)
    #get current angle between x and b
    phi = np.arccos(np.corrcoef(x, b)[0,1])
    #how much angle to shift b
    beta = phi - theta
    mag_b = np.linalg.norm(b)#adjacent length
    mag_e = np.linalg.norm(e)#opposite length
    #solve for scaling of a to the opposite side for beta
    a = (mag_b * np.tan(beta))/mag_e
    shifted_vector = a * e + b # now shift b with a*e
    return shifted_vector

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

import pickle
goforit=True      
if 'netwts' not in locals() or goforit:
    with open(top_dir + '/nets/netwts.p', 'rb') as f:    
        try:
            netwts = pickle.load(f, encoding='latin1')
        except:
            netwts = pickle.load(f)
# reshape fc layer to be spatial
netwts[5][1] = netwts[5][1].reshape((4096, 256, 6, 6))
wts_by_layer = [layer[1] for layer in netwts]

net_resp_name = 'bvlc_reference_caffenety_test_APC362_pix_width[32.0]_x_(104.0, 124.0, 11)_x_(104.0, 124.0, 11)_amp_None.nc'
da = xr.open_dataset(top_dir + '/data/responses/' + net_resp_name)['resp']
if not type(da.coords['layer_label'].values[0]) == str:
    da.coords['layer_label'].values = [thing.decode('UTF-8') for thing in da.coords['layer_label'].values]
da.coords['unit'] = range(da.shape[-1])

from more_itertools import unique_everseen
layer_num = da.coords['layer']
layer_label_ind = da.coords['layer_label'].values
split_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',]
dims = ['unit','chan', 'y', 'x']
layer_names = list(unique_everseen(layer_label_ind))
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6',]


netwtsd = {}
for layer, name in zip(wts_by_layer, layer_names):
    dim_names = dims[:len(layer.shape)]
    layer_ind = da.coords['layer_label'].values == name 
    _ =  da[..., layer_ind].coords['unit']
    netwtsd[name] = xr.DataArray(layer, dims=dims, 
           coords=[range(n) for n in np.shape(layer)])
    netwtsd[name].coords['unit'] = _
    
#%%
desired_r = 0.8
def adjust_layer_wtcov(layer, desired_r):
    layer_adj = layer.copy()
    u_da, s_da, v_da = prin_comp_maps(layer_adj)
    pc1 = u_da[..., 0]
    sign_PC = np.sign(v_da[:, 0, ...].sum(['x', 'y']))
    for i, pc, unit, sign in zip(range(len(sign_PC)), pc1, layer_adj, sign_PC):
        unit_val = unit.values.reshape(unit.shape[0], np.product(unit.shape[1:]))
        for j, pos in enumerate(unit_val.T):
            shifted_vector = adjust_to_r(sign.values*pc.values, pos, desired_r)
            unit_val[:, j] = shifted_vector
        layer_adj[i, ...] =  unit_val.reshape(unit.shape)
    return layer_adj
#opponency_da = spatial_opponency(conv2_adj) 
#%%
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ann_fn = 'bvlc_reference_caffenet'
sys.path.append('/home/dean/caffe/python')

import caffe
net_proto_name = ann_dir + 'deploy_fixing_relu_saved.prototxt'
net_wts_name = ann_dir + ann_fn + '.caffemodel'
net = caffe.Net(net_proto_name, net_wts_name, caffe.TEST)
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
print([net.params[name][0].data.shape for name in layer_names])

layer_names = ['fc6',]
for r in [  0.5, 0.6, 0.7, 0.8 ]:
    for i, name in enumerate(layer_names):
        wts = netwtsd[name]
        
        if name == 'fc6':
            adj_vals = adjust_layer_wtcov(wts, r).values
            net.params[name][0].data[...] = adj_vals.reshape((adj_vals.shape[0], np.product(adj_vals.shape[1:])))
        else:
            net.params[name][0].data[...] = adjust_layer_wtcov(wts, r).values
            
        net.save(ann_dir + 'bvlc_caffenet_reference_increase_wt_cov_fc6_'+ str(r) + '.caffemodel')
#%%
#from scipy.stats import kurtosis
#def ti_av_cov(da):
#    dims = da.coords.dims
#    #get the da in the right shape
#    if ('x' in dims) and ('y' in dims):
#        da = da.transpose('unit','shapes', 'x', 'y')
#    elif ('x' in dims):
#        da = da.transpose('unit', 'shapes', 'x')
#    elif ('y' in dims):
#        da = da.transpose('unit', 'shapes', 'y')
#        
#    #some data to store
#    ti = np.zeros(np.shape(da)[0])
#    dens = np.zeros(np.shape(da)[0])
#    nums = np.zeros(np.shape(da)[0])
#    tot_vars = np.zeros(np.shape(da)[0])
#    kurt_shapes = np.zeros(np.shape(da)[0])
#    kurt_x =  np.zeros(np.shape(da)[0])
#
#    for i, unit_resp in enumerate(da):
#        if len(unit_resp.shape)>2:
#            #unwrap spatial
#            unit_resp = unit_resp.values.reshape(unit_resp.shape[0], unit_resp.shape[1]*unit_resp.shape[2])   
#        else:
#            unit_resp = unit_resp.values
#        unit_resp = unit_resp.astype(np.float64)
#        unit_resp = unit_resp - np.mean(unit_resp, 0, keepdims=True, dtype=np.float64)
# 
#
#        cov = np.dot(unit_resp.T, unit_resp)
#        cov[np.diag_indices_from(cov)] = 0
#        numerator = np.sum(np.triu(cov))
#
#        vlength = np.linalg.norm(unit_resp, axis=0, keepdims=True)
#        max_cov = np.outer(vlength.T, vlength)
#        max_cov[np.diag_indices_from(max_cov)] = 0
#        denominator= np.sum(np.triu(max_cov))
#
#        kurt_shapes[i] = kurtosis(np.sum(unit_resp**2, 1))
#        kurt_x[i] = kurtosis(np.sum(unit_resp**2, 0))
#        den = np.sum(max_cov)
#        num = np.sum(cov)
#        dens[i] = den
#        nums[i] = num
#        tot_vars[i] = np.sum(unit_resp**2)
#        if den!=0 and num!=0:
#            ti[i] = num/den 
#    return ti, kurt_shapes, kurt_x, dens, nums, tot_vars 
#
##%%
#subsamp = 1 
#da = xr.open_dataset(top_dir + '/data/responses/bvlc_reference_caffenet_APC362_pix_width[32.0]_x_(74.0, 154.0, 21)_y_(74.0, 154.0, 21)_amp_None.nc')['resp']
#net_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc'
#da = xr.open_dataset(top_dir + '/data/responses/'+net_name)['resp'].squeeze()
#
##da = da[:,0,:,:,:]
##da.dims
#da = da.transpose('unit','shapes', 'x', 'y')
#da = da[::subsamp, ...] #subsample
#da = da.load()
#da = da - da[:, 0, :, :] #subtract off baseline
#da = da[:, 1:, ...] #get rid of baseline shape 
#
#ti_yx, kurt_shapes_yx, kurt_yx, dens, nums, tot_vars_yx = ti_av_cov(da[:, :, :, :])
#
#import pandas as pd
#opp_by_layer = []
#layer_labels = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6']
#for layer, layer_name in zip(netwts, layer_labels):
#    print(layer[1].shape)
#    if len(layer[1].shape)>2:
#        _ = xr.DataArray(layer[1], dims=['unit', 'shapes', 'x', 'y'])
#        opp = spatial_opponency(_)
#        print(len(opp))
#        opp_by_layer.append(opp)
#wt_cov = np.concatenate(opp_by_layer)
#
#non_k_var = (kurt_shapes_yx<42) * (kurt_shapes_yx>2) * (tot_vars_yx>0) 
#keys = ['layer_label', 'unit']
#coord = [da.coords[key].values for key in keys]
#index = pd.MultiIndex.from_arrays(coord, names=keys)
#resp = pd.DataFrame(np.hstack([ti_yx,]), index=index, columns=['ti',])
#layersbyunit = [[name,]*layer_wts[1].shape[0] for name, layer_wts in zip(layer_labels, netwts)]
#keys = ['layer_label',]
#index = pd.MultiIndex.from_arrays([np.concatenate(layersbyunit),], names=keys)