# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:06:55 2016

@author: dean
"""
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')

import xarray as xr
import apc_model_fit as ac
import pandas as pd
import matplotlib.ticker as mtick
try:
    import matplotlib.cm as cm
except:
    print('no plot')
from scipy.stats import kurtosis as kurtosis_sc

def kurtosis(da):
    #take xarray and return coefficient of variation
    #expects shapes X unit
    try:
        da = da - da.loc(shapes=-1)
        da = da.drop(-1, dim='shapes')
    except:
        print('no baseline, ie no shape indexed as -1')
    #da = da.transpose('shapes','unit')
    mu = da.mean('shapes')
   # k = da.reduce(kurtosis,dim='shapes')
    sig = da.reduce(np.nanvar, dim='shapes')
    k = (((da - mu)**4).sum('shapes',skipna=True)/da.shapes.shape[0])/(sig**2)
    return k

def in_rf(da, w):
    da = da.transpose('shapes','x', 'unit')
    try:
        base_line = da.sel(shapes=-1)[0]
        da = da.drop(-1, dim='shapes')
    except:
        base_line = 0
    da_bls = da - base_line#subtract off baseline
    da_var = ((da_bls)**2).sum('shapes')
    had_resp = da_var > 0
    #widest width 24
    step_width = np.diff(da_var.coords['x'].values)[1]
    #add this to the right alt, and subtract it from the left alt
    min_steps = int(np.ceil(w /step_width))
    in_rf = np.zeros(had_resp.T.values.shape)
    n_steps = len(da_var.coords['x'].values)
    rf_pos_all = []
    rf_pos = []
    beg_pos = None
    for n_unit, unit in enumerate(had_resp.T.values):

        if sum(unit)<n_steps:
            for i, x in enumerate(unit):
                if x and type(beg_pos)==type(None):
                    beg_pos = i
                elif (not x) and (type(beg_pos)!=type(None)):
                    end_pos = i-1
                    if (end_pos-beg_pos)>(min_steps*2):
                        rf_pos = rf_pos + list(range(beg_pos+min_steps, end_pos-min_steps))
                    beg_pos = None
            if x and (type(beg_pos)!=type(None)):
                end_pos = i
                if (end_pos-beg_pos)>(min_steps*2):
                    rf_pos = rf_pos + list(range(beg_pos+min_steps, n_steps-min_steps))

        else:
            rf_pos = list(range(min_steps, n_steps-min_steps))


        in_rf[n_unit, rf_pos] = 1
        rf_pos_all.append(rf_pos)
        beg_pos = None
        rf_pos = []
    return in_rf

def cross_val_SVD_TI(da, rf=None):
    from sklearn.cross_validation import KFold
    da = da.transpose('unit', 'x', 'shapes')
    try:
       da = da.drop(-1, dim='shapes')
    except:
        print('no baseline, ie no shape indexed as -1')
    if type(rf)==type(None):
        rf = np.ones(da.shape[:2])

    ti_est = []
    ti_est_all = []
    counter = 0
    da = da - da.mean('shapes')
    resp = da.values
    for unit_resp, unit_in_rf in zip(resp, rf):
        if unit_in_rf.sum()>3:
            counter = counter + 1
            if counter%100==0:
                print(counter)
            unit_resp = unit_resp[unit_in_rf.astype(bool), :]
            dr = xr.DataArray(unit_resp)
            dr = dr.dropna('dim_1',how='all')
            dr = dr.dropna('dim_0',how='all')
            unit_resp = dr.values
            loo = KFold(unit_resp.shape[0], shuffle=True, random_state=1)
            for train, test in loo:
                u, s, v = np.linalg.svd(unit_resp[train])
                ti_est = ti_est + [sum((np.dot(v[0], unit_resp[test].T))**2),]
            tot_var = (unit_resp**2).sum()
            ti_est = np.sum(ti_est)/tot_var
            ti_est_all.append(ti_est)
        else:
            ti_est_all.append(np.nan) 
        ti_est = []
    return ti_est_all

def SVD_TI(da, rf=None):
    da = da.transpose('unit', 'x', 'shapes')
    try:
       da = da.drop(-1, dim='shapes')
    except:
        print('no baseline, ie no shape indexed as -1')

    if type(rf)==type(None):
        rf = np.ones(da.shape[:2])
        no_rf = True
    else:
        no_rf = False

    ti_est_all = []
    counter = 0
    da = da - da.mean('shapes')
    resp = da.values

    for unit_resp, unit_in_rf in zip(resp, rf):
        if counter%100 == 0:
            print(counter)
        counter = counter + 1

        if sum(unit_in_rf)>2:
            if not no_rf:
                 unit_resp = unit_resp[unit_in_rf.astype(bool), :]
            dr = xr.DataArray(unit_resp)
            dr = dr.dropna('dim_1',how='all')
            dr = dr.dropna('dim_0',how='all')
            unit_resp = dr.values
            singular_values = np.linalg.svd(unit_resp, compute_uv=False)
            frac_var = (singular_values[0]**2)/(sum(singular_values**2))
            ti_est_all.append(frac_var)
        else:
            ti_est_all.append(np.nan)
    return ti_est_all

def cnn_measure_to_pandas(da, measures, measure_names):
    keys = ['layer_label', 'unit']
    coord = [da.coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    pda = pd.DataFrame(np.array(measures).T, index=index, columns=measure_names)


    return pda

 
def ti_av_cov(da, rf=None):
    da = da.transpose('unit', 'x', 'shapes')
    try:
        da = da - da.loc(shapes=-1)
        da = da.drop(-1, dim='shapes')
    except:
        print('no baseline, ie no shape indexed as -1')
    if type(rf)==type(None):
        rf = np.ones(da.shape[:2])
        no_rf = True
    else:
        no_rf = False

    ti_est_all = []
    counter = 0
    da = da - da.mean('shapes')
    resp = da.values

    for unit_resp, unit_in_rf in zip(resp, rf):
        if counter%100 == 0:
            print(counter)
        counter = counter + 1

        if sum(unit_in_rf)>2:
            if not no_rf:
                 unit_resp = unit_resp[unit_in_rf.astype(bool), :]
            dr = xr.DataArray(unit_resp)
            dr = dr.dropna('dim_1',how='all')
            dr = dr.dropna('dim_0',how='all')
            unit_resp = dr.values
            cov = np.dot(unit_resp, unit_resp.T)
            cov[np.diag_indices_from(cov)] = 0
            numerator = np.sum(np.triu(cov))
            vlength = np.linalg.norm(unit_resp, axis=1)
            max_cov = np.outer(vlength.T, vlength)
            max_cov[np.diag_indices_from(max_cov)] = 0
            denominator= np.sum(np.triu(max_cov))
            frac_var = numerator/denominator
            ti_est_all.append(frac_var)
        else:
            ti_est_all.append(np.nan)
    return ti_est_all


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

def ti_in_rf(resp, stim_width=None):
    try:
        base_line = resp.sel(shapes=-1)[0]
        resp = resp.drop(-1, dim='shapes')
    except:
        base_line = 0
    resp = resp - base_line#subtract off baseline
    dims = resp.coords.dims
    if stim_width == None:
        if ('x' in dims) and ('y' in dims):
            resp = resp.transpose('unit','shapes', 'x', 'y')
            resp_unrolled = resp.values.reshape(resp.shape[:2] + (np.product(resp.shape[-2:]),))
            ti= []
            for a_resp in  resp_unrolled:
                ti.append(norm_cov(a_resp)) 
        elif ('x' in dims) or ('y' in dims):
            if 'x' in dims:
                resp = resp.transpose('unit', 'shapes', 'x')
                
            elif 'y' in dims:
                resp = resp.transpose('unit', 'shapes', 'y')
            ti = []
            for a_resp in resp.values:
                ti.append(norm_cov(a_resp))
    else:
        if ('x' in dims) and ('y' in dims):
    
            resp = resp.transpose('unit','shapes', 'x', 'y')
            
            x = resp.coords['x'].values
            y = resp.coords['y'].values
            
            x_grid = np.tile(x, (len(y), 1)).ravel()
            y_grid = np.tile(y[:, np.newaxis], (1, len(x))).ravel()
            
            x_dist = x_grid[:, np.newaxis] - x_grid[:, np.newaxis].T
            y_dist = y_grid[:, np.newaxis] - y_grid[:, np.newaxis].T
            
            dist_mat = (x_dist**2 + y_dist**2)**0.5
            stim_in = dist_mat<=(stim_width*1.)
            rf = (resp**2).sum('shapes')>0
            rf[..., :, -1] = False
            rf[..., :, 0] = False
            rf[..., 0, :] = False
            rf[..., -1, :] = False
            rf = rf.values.reshape((rf.shape[0],) + (np.product(rf.shape[1:]),))
            in_spots = stim_in.sum(0)
            overlap = np.array([a_rf * stim_in for a_rf in rf]).sum(-1)
            in_rf = overlap == in_spots[np.newaxis,:]
            
            resp_unrolled = resp.values.reshape(resp.shape[:2] + (np.product(resp.shape[-2:]),))
            ti= []
            for an_in_rf, a_resp in zip(in_rf, resp_unrolled):
                if np.sum(an_in_rf)>2:
                    ti.append(norm_cov(a_resp[..., an_in_rf.squeeze()]))
                else:
                    ti.append(np.nan)
            
        elif ('x' in dims) or ('y' in dims):
            if 'x' in dims:
                resp = resp.transpose('unit', 'shapes', 'x')
                pos = resp.coords['x'].values
                
            elif 'y' in dims:
                resp = resp.transpose('unit', 'shapes', 'y')
                pos = resp.coords['y'].values
        
            pos_dist = pos[:, np.newaxis] - pos[:, np.newaxis].T #matrix of differences
            dist_mat = (pos_dist**2)**0.5 #matrix of distances
            stim_in = dist_mat<=(stim_width*1.)#all positions you need to check if responded
            rf = (resp**2).sum('shapes')>0
            #hackish way to make sure test pos is far enough from edge
            #for example if you test two positions close to each other, all adjacent stim
            #are activated but all are on edge, so can't be sure.
            rf[..., 0] = False
            rf[..., -1] = False
            in_rf = rf.copy()
            in_spots = stim_in.sum(0)
            #after overlap only the intersection of stim_in
            #and rf exists so if it is any less then stim_in then not all stim_in points
            #were activated.
            ti = []
            for i, an_rf in enumerate(rf):
                overlap = np.sum(an_rf.values[:, np.newaxis] * stim_in, 0)
                in_pos = overlap == in_spots
                in_rf[i] = in_pos
                
            for an_in_rf, a_resp in zip(in_rf.values, resp.values):
                if np.sum(an_in_rf)>2:
                    ti.append(norm_cov(a_resp[..., an_in_rf.squeeze()]))
                else:
                    ti.append(np.nan)
        
    resp_av_cov_da = xr.DataArray(ti, coords=resp.coords['unit'].coords)  
    return resp_av_cov_da

    
def norm_avcov_iter(x, subtract_mean=True):
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 1, keepdims=True)
    diag_inds = np.triu_indices(x.shape[-1], k=1)
    numerator = [np.sum(np.dot(unit.T, unit)[diag_inds]) for unit in x]
    
    vnrm = np.linalg.norm(x, axis=1, keepdims=True)
    denominator = [np.sum(np.multiply(unit.T, unit)[diag_inds]) for unit in vnrm]    
    norm_cov = np.array(numerator)/np.array(denominator)
    norm_cov[np.isnan(norm_cov)] = 0
    
    return norm_cov


def spatial_weight_normcov(netwtsd):
    unit_coords = xr.concat([netwtsd[key].coords['unit'] 
                            for key in netwtsd.keys()], 'unit').coords
    netwts_list = []
    for key in netwtsd:
        netwt = netwtsd[key].transpose('unit', 'chan', 'y', 'x').values       
        netwts_list.append(netwt)
    
    av_cov_list = []
    for layer_wt in netwts_list:
        o_shape = np.shape(layer_wt)
        ravel_space = o_shape[:2] + (np.product(o_shape[2:]),)
        av_cov_list.append(norm_avcov_iter(layer_wt.reshape(ravel_space), subtract_mean=True))
    
    av_cov = np.concatenate(av_cov_list)
    av_cov_da = xr.DataArray(av_cov, unit_coords)    
    return av_cov_da


def kurtosis_da(resp):
    dims = resp.coords.dims   
    
    if ('x' in dims) or ('y' in dims):
        if ('x' in dims) and ('y' in dims):
            resp = resp.transpose('unit', 'shapes', 'x', 'y')
            stim_resp = np.array([(unit**2).sum((1, 2)) for unit in resp.values])
            pos_resp = np.array([(unit**2).sum(0).ravel() for unit in resp.values])
        elif ('x' in dims):
            resp = resp.transpose('unit', 'shapes', 'x')
            stim_resp = np.array([(unit**2).sum((1)) for unit in resp.values])
            pos_resp = np.array([(unit**2).sum(0).ravel() for unit in resp.values])
        elif ('y' in dims):
            resp = resp.transpose('unit', 'shapes', 'y')
            stim_resp = np.array([(unit**2).sum((1)) for unit in resp.values])
            pos_resp = np.array([(unit**2).sum(0).ravel() for unit in resp.values])
        k_stim = kurtosis_sc(stim_resp, axis=1, fisher=False)
        k_pos = kurtosis_sc(pos_resp, axis=1, fisher=False)
        return k_pos, k_stim
    else:
        resp = resp.transpose('unit', 'shapes')
        k_stim = kurtosis_sc(resp.values, axis=1, fisher=False)
        
    return k_stim

def tot_var(resp):
    dims = resp.coords.dims   
    if ('x' in dims) and ('y' in dims):
        resp = resp.transpose('unit','shapes', 'x', 'y')
    elif ('x' in dims):
        resp = resp.transpose('unit', 'shapes', 'x')
    elif ('y' in dims):
        resp = resp.transpose('unit', 'shapes', 'y')
        
    pwr = np.array([(unit**2).sum() for unit in resp.values])
    return pwr
    



#def nice_axes(axes, xticks=None, yticks=None, nxticks=5, nyticks=2):
#
#    for i, an_axes in enumerate(axes):
#        if i==len(axes)-1:
#            if yticks==None:
#                an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
#                an_axes.set_yticks([])
#            else:
#                an_axes.set_yticks(yticks)
#                an_axes.set_yticks([])
#            if xticks==None:
#               an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
#            else:
#                an_axes.set_xticks(xticks)
#                an_axes.xaxis.set_tick_params(length=0)
#                an_axes.yaxis.set_tick_params(length=0)
#                an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
#            an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
#        else:
#            an_axes.set_xticks([])
#            an_axes.set_yticks([])
#
#def stacked_hist_layers(cnn, logx=False, logy=False, xlim=None, maxlim=False,
#                        bins=100, cumulative=False, normed=False):
#    layers = cnn.index.get_level_values('layer_label').unique()
#    if logx:
#        cnn = np.log(cnn.dropna())
#    if maxlim:
#        xlim = [np.min(cnn.dropna().values), np.max(cnn.dropna().values)]
#    for i, layer in enumerate(layers):
#        plt.subplot(len(layers), 1, i+1)
#        vals = cnn.loc[layer].dropna().values.flatten()
#
#
#        plt.hist(vals, log=logy, bins=bins, histtype='step',
#                 range=xlim, normed=normed, cumulative=cumulative)
#        if cumulative:
#            plt.ylim(0,1.1)
#        plt.plot([np.median(vals),]*2, np.array(plt.gca().get_ylim()), color='red')
#        plt.xlim(xlim)
#        plt.gca().set_ylabel(layer, ha='right', rotation=0, labelpad=25)
#        plt.gca().yaxis.set_label_position("right")
#
#    if logx:
#        plt.xlabel('log')
#    nice_axes(plt.gcf().axes)