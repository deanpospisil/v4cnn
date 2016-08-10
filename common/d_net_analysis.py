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
    sig = da.reduce(np.var, dim='shapes')
    k = (((da - mu)**4).sum('shapes')/da.shapes.shape[0])/(sig**2)
    return k

def in_rf(da, w):
    da = da.transpose('shapes','x', 'unit')
    base_line = da.sel(shapes=-1)[0]
    da = da.drop(-1, dim='shapes')

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

def cross_val_SVD_TI(da, rf):
    from sklearn.cross_validation import LeaveOneOut
    da = da.transpose('unit', 'x', 'shapes')
    ti_est = []
    ti_est_all = []
    counter = 0
    da = da - da.mean('shapes')
    resp = da.values
    for unit_resp, unit_in_rf in zip(resp, rf):
        if unit_in_rf.sum()>1:
            counter = counter + 1
            print(counter)
            unit_resp = unit_resp[unit_in_rf.astype(bool), :]
            loo = LeaveOneOut(unit_resp.shape[0])
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
    if type(rf)==type(None):
        rf = np.ones(da.shape[1:])
        no_rf = True
    else:
        no_rf = False

    ti_est_all = []
    counter = 0
    da = da - da.mean('shapes')
    resp = da.values
    for unit_resp, unit_in_rf in zip(resp, rf):
        counter = counter + 1
        print(counter)
        if sum(unit_in_rf)>2:
            if no_rf:
                unit_resp = unit_resp
            else:
                 unit_resp = unit_resp[unit_in_rf.astype(bool), :]

    
            singular_values = np.linalg.svd(unit_resp, compute_uv=False)
            frac_var = (singular_values[0]**2)/(sum(singular_values**2))
            ti_est_all.append(frac_var)
        else:
            ti_est_all.append(np.nan)
    return ti_est_all

def cnn_measure_to_pandas(da, measures, measure_names):
    keys = ['layer_label', 'layer_unit']
    coord = [da['unit'].coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    pda = pd.DataFrame(np.array(measures).T, index=index, columns=measure_names)
    return pda

cnn_name = 'APC362_scale_1_pos_(-99, 96, 66)bvlc_reference_caffenet'
da = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp']
da = da.sel(unit=slice(1000, 5000)).load().squeeze()
#rf = in_rf(da, w=24)
#cv_ti = cross_val_SVD_TI(da, rf)
#ti = SVD_TI(da, rf)
ti_orf = SVD_TI(da)



da_0 = da.sel(x=0)
k = list(kurtosis(da_0).values)




#measure_names=['ti','cv_ti','k']
#measures = [ti, cv_ti, k]
#pda = cnn_measure_to_pandas(da_0, measures, measure_names)

measure_names=['ti','ti_orf', 'k']
measures = [ti,ti_orf,  k]
pda = cnn_measure_to_pandas(da_0, measures, measure_names)

type_change = np.where(np.diff(da.coords['layer'].values))[0]
type_label = da.coords['layer_label'].values[type_change].astype(str)

plt.scatter(range(len(ti)),ti)
plt.xticks(type_change, type_label, rotation='vertical', size = 'small')

pda.plot.scatter('ti')