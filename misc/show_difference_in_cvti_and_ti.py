# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:33:29 2016

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
import pickle
import xarray as xr
import apc_model_fit as ac
import pandas as pd
import matplotlib.pyplot as plt

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


with open(top_dir + 'data/an_results/' + 'fixed_relu_saved_24_30_pix.p', 'rb') as f:
    pan = pickle.load(f)
    pda = pan['24']

k_thresh = 40

kt_pda = pda[pda['k']<k_thresh]
most_dif = (kt_pda['ti'] - kt_pda['cv_ti']).dropna().argsort()


colors = ['r', 'g', 'b', 'm', 'k']
cnn_names =['APC362_deploy_fixing_relu_saved.prototxt_fixed_even_pix_width[24.0, 48.0]_pos_(64.0, 164.0, 51)bvlc_reference_caffenet',
'APC362_deploy_fixing_relu_saved.prototxt_shuffle_fixed_even_pix_width[24, 30.0]_pos_(64.0, 164.0, 51)bvlc_caffenet_reference_shuffle']
if 'da' not in locals():
    da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].squeeze().isel(scale=0).load()
unit_ind = (kt_pda['ti'] - kt_pda['cv_ti']).argmax()
big_dif = da.sel(unit=most_dif[-10::-1])
#plt.figure()
#big_dif.plot()
#plt.title(str(kt_pda.loc[unit_ind][['ti']]) + '  '+str(kt_pda.loc[unit_ind][['cv_ti']]) )
#plt.tight_layout()
inrf = in_rf(big_dif, 24)
ti_est_all = cross_val_SVD_TI(big_dif, rf=inrf)