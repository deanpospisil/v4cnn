# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:04:12 2016

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
                        bins=100, cumulative=False, normed=False,
                        extra_subplot=False, title=None, layers=None, color=None):
    if layers==None:
        layers = cnn.index.get_level_values('layer_label').unique()
    if logx:
        cnn = np.log10(cnn.dropna())
        xlim = np.log10(xlim)
    if maxlim:
        xlim = [np.min(cnn.dropna().values), np.max(cnn.dropna().values)]
    if color==None:
        c='b'
    n_subplot = len(layers)+extra_subplot
    for i, layer in enumerate(layers):
        plt.subplot(n_subplot, 1, i+1)

        if title!=None and i==0:
            plt.title(title)
        try:
            vals = cnn.loc[layer].dropna().values.flatten()
            plt.hist(vals, log=logy, bins=bins, histtype='step',
                     range=xlim, normed=normed, cumulative=cumulative,color=color)
            if cumulative:
                plt.ylim(0,1.1)
            plt.plot([np.median(vals),]*2, np.array(plt.gca().get_ylim()), color=color)
        except:
            print('no data ' + layer)
        plt.xlim(xlim)

        plt.gca().set_ylabel(layer, ha='right', rotation=0, labelpad=25)
        plt.gca().yaxis.set_label_position("right")

    if logx:
        plt.xlabel('log')

    nice_axes(plt.gcf().axes)

def process_V4(v4_resp_apc, v4_resp_ti, dmod):
    cv_ti = cross_val_SVD_TI(v4_resp_ti, rf=None)
    ti = SVD_TI(v4_resp_ti, rf=None)
    apc = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), dmod.chunk({}), fit_over_dims=None, prov_commit=False)
    k_apc = list(kurtosis(v4_resp_apc).values)
    k_ti = list(kurtosis(v4_resp_ti.mean('x')).values)

    keys = ['layer_label', 'unit']
    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(ti)),np.arange(len(ti))], names=keys)
    v4pdti  = pd.DataFrame(np.array([ti, cv_ti, k_ti]).T, index=index, columns=['ti', 'cv_ti', 'k'])

    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(apc)),np.arange(len(apc))], names=keys)
    v4pdapc  = pd.DataFrame(np.array([apc.values, k_apc]).T, index=index, columns=['apc', 'k'])
    v4 = pd.concat([v4pdti,v4pdapc])
    return v4
colors = ['r', 'g', 'b', 'm', 'k']
cnn_names =['APC362_deploy_fixing_relu_saved.prototxt_fixed_even_pix_width[24.0, 48.0]_pos_(64.0, 164.0, 51)bvlc_reference_caffenet',
'APC362_deploy_fixing_relu_saved.prototxt_shuffle_fixed_even_pix_width[24, 30.0]_pos_(64.0, 164.0, 51)bvlc_caffenet_reference_shuffle']

da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=0)
da = da.sel(unit=slice(0,None,1)).squeeze()
middle = np.round(len(da.coords['x'])/2.)
da_0 = da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)])

indexes = np.unique(da_0.coords['layer_label'].values, return_index=True)[1]
layer_label = [da_0.coords['layer_label'].values[index] for index in sorted(indexes)]
indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
layer_label = [da.coords['layer_label'].values[index] for index in sorted(indexes)]

figure_folder = top_dir + 'analysis/figures/images/'
k_thresh = 40
names = ['24', '30']
name = names[0]


'''
v4ness_nms = ['fixed_relu_saved_24_30_pix.p','null_fixed_relu_saved_24_30_pix.p',
          'null_shuffle_fixed_relu_saved_24_30_pix.p' ]

v4ness_list = []
for nms in v4ness_nms:
    with open(top_dir + 'data/an_results/' + nms, 'rb') as f:
        pan = pickle.load(f)
        pda = pan[name]
    v4ness_list.append(pda.reindex(layer_label, level='layer_label'))


if 'dmod' not in locals():
    fn = top_dir + 'data/models/' + 'apc_models_362.nc'
    dmod = xr.open_dataset(fn, chunks={'models': 50, 'shapes': 370}  )['resp'].load()
    v4_name = 'V4_362PC2001'
    v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()
    alt_v4 = process_V4(v4_resp_apc, v4_resp_ti, dmod)

    #shuffle
    v4_resp_apc_null = v4_resp_apc.copy()
    v4_resp_ti_null = v4_resp_ti.copy()

    for  x in range(len(v4_resp_ti_null.coords['x'])):
        for unit in range(len(v4_resp_ti_null.coords['unit'])):
            not_null = ~v4_resp_ti_null[unit,x,:].isnull()
            v4_resp_ti_null[unit,x, not_null] = np.random.permutation(v4_resp_ti[unit,x,not_null].values)

    for unit in range(len(v4_resp_apc_null.coords['unit'])):
        v4_resp_apc_null[:,unit] = np.random.permutation(v4_resp_apc[:,unit].values)

    null_v4 = process_V4(v4_resp_apc_null, v4_resp_ti_null, dmod)




v4ness_list.append(alt_v4)
v4ness_list.append(null_v4)
keys=['alt_net', 'null_resp','null_net', 'alt_v4', 'null_v4']
v4ness = pd.concat(v4ness_list, keys=keys)

v4ness['apc'] = v4ness['apc']**2

title = 'Fraction with RF'
plt.close('all')
plt.figure(figsize=(12,5))
(v4ness['in_rf']>2).groupby(level=[0,1],sort=False).mean().plot(kind='bar')
plt.title('Fraction with RF');plt.ylim(0,1);plt.grid();plt.ylabel('Mean no response')
plt.tight_layout()
plt.savefig(figure_folder + title +'.png')


title= 'cv_ti Median by layer'
plt.figure(figsize=(12,5))
medians = v4ness[v4ness['k']<k_thresh].groupby(level=[0,1],sort=False).median()
medians['cv_ti'].plot(kind='bar')
plt.title(title);plt.ylim(0,1);plt.grid();plt.ylabel('Median CV TI Frac Var')
plt.tight_layout()
plt.savefig(figure_folder + title +'.png')


title ='APC median by layer'
plt.figure(figsize=(12,5))
medians['apc'].plot(kind='bar')
plt.title(title);
plt.yticks(np.linspace(0,1,22));
plt.ylim(0, .6);plt.grid();plt.ylabel('Median APC Frac Var');plt.tight_layout()
plt.savefig(figure_folder + title +'.png')

plt.figure(figsize=(12,5))
v4ness['k'].groupby(level=[0,1],sort=False).median().plot(kind='bar', log=True)
plt.title('k');plt.grid();plt.ylabel('Median k');plt.tight_layout()
title = 'RF_size by layer'
plt.figure(figsize=(12,5))
v4ness['in_rf'].groupby(level=[0,1],sort=False).median().plot(kind='bar', log=False)
plt.title(title);plt.grid();plt.ylabel('Median rf size');plt.tight_layout()
plt.savefig(figure_folder + title +'.png')

#%%
plt.figure()
colors = ['r', 'g', 'b', 'm', 'k']
sub_keys = keys[0:3]
title = 'v4ness distribution'
for i, key in enumerate(sub_keys):
    df=v4ness.loc[key]
    df = df[df['k']<k_thresh]
    plt.scatter(df['ti'], df['apc'], s=5, c=colors[i], marker='.', alpha=1,linewidths=0)
plt.legend(sub_keys,markerscale=10)
plt.xlabel('TI frac var');plt.ylabel('apc frac var');plt.xlim(0,1);plt.ylim(0,1);
plt.savefig(figure_folder + title +'.png')


plt.figure()
title = 'fc7 best in null_net'
df = v4ness.loc['null_net']
df = df[df['k']<k_thresh]
plt.scatter(df['ti'], df['apc'], s=5, c='b', marker='.', alpha=1, linewidths=0)
plt.scatter(df.loc['fc7']['ti'], df.loc['fc7']['apc'], s=12, c='r', marker='.', alpha=1, linewidths=0)
plt.legend(['all null_net', 'fc7 null_net'], markerscale=10)
plt.xlabel('TI frac var');plt.ylabel('apc frac var');plt.xlim(0,1);plt.ylim(0,1);
plt.savefig(figure_folder + title +'.png')

#%%
plt.close('all')
plt.figure(figsize=(6,12))
sub_keys = keys[0:3]
title = 'k: alt_net:red null_net:blue null_resp:green'
for i, key in enumerate(sub_keys):
    df=v4ness.loc[key]
    stacked_hist_layers(df['k'].dropna(),
                    title=title, logx=False, logy=True, xlim=[0,370],
                     maxlim=False, bins=100, layers=layer_label, color=colors[i])
plt.savefig(figure_folder + title +'.png')


#%%
plt.figure(figsize=(6,12))
sub_keys = keys[0:3]
title = 'TI_frac_var: alt_net:red null_net:blue null_resp:green'
for i, key in enumerate(sub_keys):
    df=v4ness.loc[key]
    df = df[df['k']<k_thresh]
    stacked_hist_layers(df['cv_ti'].dropna(),
                    title=title, logx=False, logy=True, xlim=[0,1],
                     maxlim=False, bins=100, layers=layer_label, color=colors[i])
plt.savefig(figure_folder + title +'.png')

 #%%
plt.figure(figsize=(6,12))
sub_keys = keys[0:3]
title = 'APC_frac_var: alt_net:red null_net:blue null_resp:green'
for i, key in enumerate(sub_keys):
    df=v4ness.loc[key]
    df = df[df['k']<k_thresh]
    stacked_hist_layers(df['apc'].dropna(),
                    title=title, logx=False, logy=True, xlim=[0,1],
                     maxlim=False, bins=100, layers=layer_label, color=colors[i])
plt.savefig(figure_folder + title +'.png')

#%%
plt.figure(figsize=(6,12))
kt_v4ness = v4ness
kt_v4ness =kt_v4ness[kt_v4ness['k']<k_thresh]
dists = ((1-v4ness['apc'])**2 + (1-v4ness['cv_ti'])**2)**0.5
sub_keys = keys[0:3]
title = 'v4_distance: alt_net:red null_net:blue null_resp:green'
for i, key in enumerate(sub_keys):
    df=dists.loc[key]
    stacked_hist_layers(df.dropna(),
                    title=title, logx=False, logy=True, xlim=[0,2**0.5],
                     maxlim=False, bins=100, layers=layer_label, color=colors[i])
plt.savefig(figure_folder + title +'.png')

#%%
center_resps = []
title = 'response_power_bl_sub_over_pos'
da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[1] + '.nc')['resp'].isel(scale=0)
da = da - da[0]
rf = (da**2).sum('shapes')
rf_null = rf / rf.max('x')
center_resps.append(da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)]))

da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=0)
da = da - da[0]
rf = (da**2).sum('shapes')
rf_alt = rf / rf.max('x')
center_resps.append(da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)]))

type_change = np.where(np.diff(da.coords['layer'].values))[0]
type_label = da.coords['layer_label'].values[type_change].astype(str)

plt.figure(12,12)
plt.subplot(211)
rf_null.plot()
plt.xticks(type_change, type_label, rotation='vertical', size = 'small')
plt.subplot(212)
rf_alt.plot()
plt.xticks(type_change, type_label, rotation='vertical', size = 'small')
plt.tight_layout()
plt.savefig(figure_folder + title +'.png')

#%%
center_resps = []

da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=0)
da = da - da[0]
center_resps.append(da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)]))

da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[1] + '.nc')['resp'].isel(scale=0)
da = da - da[0]
center_resps.append(da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)]))

df_center_resp = []
for resp in center_resps:
    keys = ['layer_label', 'unit']
    coord = [da.coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    pda = pd.DataFrame(resp.squeeze().values.T, index=index)
    df_center_resp.append(pda)
title = 'resp_dist_shuffle_nonshuffle'
for i, resp in enumerate(df_center_resp):
    stacked_hist_layers(resp.dropna(),
                    title=title, logx=False, logy=True, xlim=None,
                     maxlim=False, bins=100, layers=layer_label, color=colors[i])
'''