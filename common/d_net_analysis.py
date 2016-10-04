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
import matplotlib
from matplotlib.ticker import FuncFormatter

import xarray as xr
import apc_model_fit as ac
import pandas as pd
import matplotlib.ticker as mtick
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except:
    print('no plot')

def kurtosis(da):
    #take xarray and return coefficient of variation
    #expects shapes X unit
    da = da.transpose('shapes','unit')
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
def tick_format_d(x, pos):
    if x==0:
        return('0')
    else:
        if x==1:
            return(str(x).split('.')[0])
        else:
            return(np.round(x, 2))


def nice_axes(axes, xticks=None, yticks=None, nxticks=5, nyticks=2):

    for i, an_axes in enumerate(axes):
        if i==len(axes)-1:
            if yticks==None:
                an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
                an_axes.set_yticks([])
            else:
                an_axes.set_yticks(yticks)
                an_axes.set_yticks([])
            if xticks==None:
               an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
            else:
                an_axes.set_xticks(xticks)
                an_axes.xaxis.set_tick_params(length=0)
                an_axes.yaxis.set_tick_params(length=0)
                an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
            an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
        else:
            an_axes.set_xticks([])
            an_axes.set_yticks([])

def stacked_hist_layers(cnn, logx=False, logy=False, xlim=None, maxlim=False,
                        bins=100, cumulative=False, normed=False):
    layers = cnn.index.get_level_values('layer_label').unique()
    if logx:
        cnn = np.log(cnn.dropna())
    if maxlim:
        xlim = [np.min(cnn.dropna().values), np.max(cnn.dropna().values)]
    for i, layer in enumerate(layers):
        plt.subplot(len(layers), 1, i+1)
        vals = cnn.loc[layer].dropna().values.flatten()


        plt.hist(vals, log=logy, bins=bins, histtype='step',
                 range=xlim, normed=normed, cumulative=cumulative)
        if cumulative:
            plt.ylim(0,1.1)
        plt.plot([np.median(vals),]*2, np.array(plt.gca().get_ylim()), color='red')
        plt.xlim(xlim)
        plt.gca().set_ylabel(layer, ha='right', rotation=0, labelpad=25)
        plt.gca().yaxis.set_label_position("right")

    if logx:
        plt.xlabel('log')
    nice_axes(plt.gcf().axes)


import pickle

measure_list =[ 'apc', 'ti', 'ti_orf', 'cv_ti', 'k', 'in_rf', 'no_response_mod']
#measure_list =['ti', 'k', 'inrf', 'no_response_mod']
fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models': 50, 'shapes': 370}  )['resp']
cnn_names =['']


pdas = []
cnns = [xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=1) , ]
null=True
widths = [30.,]
for w, da in zip(widths,cnns):
    print(w)
    np.random.seed(1)
    da = da.sel(unit=slice(0, None, None)).load().squeeze()
    if null:
        for  x in range(len(da.coords['x'])):
            print(x)
            for unit in range(len(da.coords['unit'])):
                da[1:, x, unit] = np.random.permutation(da[1:,x,unit].values)
    print(1)
    da_0 = da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)])
    rf = in_rf(da, w=w)
    measures = []
    if 'apc' in measure_list:	
        measures.append(ac.cor_resp_to_model(da_0.chunk({'shapes': 370}), dmod, fit_over_dims=None, prov_commit=False).values)
    if 'ti' in measure_list:
        measures.append(SVD_TI(da, rf))
    if 'ti_orf' in measure_list:
        measures.append(SVD_TI(da))
    if 'cv_ti' in measure_list:
        measures.append(cross_val_SVD_TI(da, rf))
    if 'k' in measure_list:		
        measures.append(list(kurtosis(da_0).values))
    if 'in_rf' in measure_list:
        measures.append(np.sum(rf,1))
    if 'no_response_mod' in measure_list:
        measures.append((((da-da.mean('shapes'))**2).sum(['shapes','x'])==0).values)

    keys = ['layer_label', 'unit']
    coord = [da_0.coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    pda = pd.DataFrame(np.array(measures).T, index=index, columns=measure_list)
    pdas.append(pda)
d = {key: value for (key, value) in zip(['24','30' ], pdas)}
pan = pd.Panel(d)
#pan.to_pickle(top_dir + 'data/an_results/fixed_relu_saved_24_30_pix.p')
pan.to_pickle(top_dir + 'data/an_results/null_fixed_relu_saved_24_30_pix.p')
#pan.to_pickle(top_dir + 'data/an_results/null_shuffle_fixed_relu_saved_24_30_pix.p')

'''
type_change = np.where(np.diff(da.coords['layer'].values))[0]
type_label = da.coords['layer_label'].values[type_change].astype(str)
plt.xticks(type_change, type_label, rotation='vertical', size = 'small')
plt.figure()
k_thresh = 40
plt.scatter(pda[pda['k']<k_thresh]['cv_ti'], pda[pda['k']<40]['ti'], alpha=1, s=2)
plt.plot([0,1],[0,1])
#plt.close('all')
plt.figure()
stacked_hist_layers(pda[pda['k']<k_thresh]['cv_ti'].dropna(), logx=False, logy=False, xlim=[0,1], maxlim=False, bins=100)
plt.suptitle('cv ti in rf')
plt.figure()
stacked_hist_layers(pda[pda['k']<k_thresh]['ti_orf'].dropna(), logx=False, logy=False, xlim=[0,1], maxlim=False, bins=100)
plt.suptitle('ti all pos')
plt.figure()
stacked_hist_layers(pda[pda['k']<k_thresh]['ti'].dropna(), logx=False, logy=False, xlim=[0,1], maxlim=False, bins=100)
plt.suptitle('ti in rf')
'''
