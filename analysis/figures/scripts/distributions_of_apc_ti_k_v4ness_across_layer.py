# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:41:01 2016

@author: dean
"""

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


cnn_names =['APC362_deploy_fixing_relu_saved.prototxt_fixed_even_pix_width[24.0, 48.0]_pos_(64.0, 164.0, 51)bvlc_reference_caffenet' ]
pdas = []
da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=0)
da_0 = da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)])


if 'dmod' not in locals():
    fn = top_dir + 'data/models/' + 'apc_models_362.nc'
    dmod = xr.open_dataset(fn, chunks={'models': 50, 'shapes': 370}  )['resp'].load()

# now I need to tack on V4
v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()

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
figure_folder = top_dir + 'analysis/figures/images/'
import pickle
plt.close('all')
with open(top_dir + 'data/an_results/fixed_relu_saved_24_48_pix.p', 'rb') as f:
    pan = pickle.load(f)
names = ['24', '48']
df = pan['24']
df = df[df.index.get_level_values('layer_label')!='prob']
all_layers = pd.concat([df, v4])

plt.figure()
k_thresh = 30
stacked_hist_layers((all_layers[all_layers['k']<k_thresh])['cv_ti'].dropna(), logx=False, logy=False, xlim=[0,1], maxlim=False, bins=100)
plt.suptitle('3-fold Cross-Validated TI measured in receptive field.' )
plt.xlabel('Fraction Variance Explained by TI Model')
plt.savefig(figure_folder + 'ti_frac_var_24pix.eps')

plt.figure()
stacked_hist_layers((((all_layers[all_layers['k']<k_thresh])['apc'])**1).dropna(),
                    logx=False, logy=False, xlim=[0,1], maxlim=False, bins=100)
plt.suptitle('Fit to APC model measured in center of receptive field.' )
plt.xlabel('Correlation with APC Model')
plt.savefig(figure_folder + 'apc_correlation_24pix.eps')


plt.figure(figsize=(7,12))
stacked_hist_layers(((all_layers['k'])).dropna(), logx=False, logy=True, xlim=[0,370], maxlim=False, bins=100)
plt.suptitle('Sparsity of Layers.' )
plt.xlabel('Kurtosis')
plt.savefig(figure_folder + 'sparsity_loghist.eps')

kts = all_layers[all_layers['k']<k_thresh]
kts['cv_ti'] = kts['cv_ti']**2
kts_c = kts[kts.index.get_level_values('layer_label')!='v4']
kts_v = kts[kts.index.get_level_values('layer_label')=='v4']
ti_m_prob, ti_m_value = np.histogram(kts_v['cv_ti'].dropna(), normed=True, bins=20)
apc_m_prob, apc_m_value  = np.histogram(kts_v['apc'].dropna(), normed=True, bins=20)

joint_v4ness = apc_m_prob.reshape(1,20)*ti_m_prob.reshape(20,1)
apc_val, ti_val = np.meshgrid(apc_m_value[1:], ti_m_value[1:])
dist_v4 = ((1- apc_val)**2 + (1- ti_val)**2)**0.5
plt.figure(figsize=(7,3))
n, bins, patches = plt.hist(dist_v4.ravel(), weights=joint_v4ness.ravel(),
                            bins=100, range=[0,1], normed=True, histtype='step',
                            cumulative=True)
plt.close('all')
plt.figure(figsize=(7,3))
plt.gca().xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
plt.hist(dist_v4.ravel(), weights=joint_v4ness.ravel(), bins=20,
                            range=[0, 2**0.5], normed=True, histtype='step', cumulative=False)
plt.plot([bins[n>.5][0],]*2, np.array(plt.gca().get_ylim()), color='red')
plt.ylim(plt.gca().get_ylim())
plt.xlim([0,2**0.5],)

plt.title('Assume independence of TI and APC measurements in V4.')
plt.xlabel('Distance from "perfect" V4: APC = 1, TI = 1.')
nice_axes(plt.gcf().axes, xticks=None, yticks=None, nxticks=5, nyticks=2)
plt.tight_layout()
plt.savefig(figure_folder + 'v4ness_ofV4_assuming_independence.eps')


plt.close('all')
plt.figure(figsize=(7,3))
plt.gca().xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
plt.hist(dist_v4.ravel(), weights=joint_v4ness.ravel(),bins=50,
                            range=[0, 2**0.5], normed=True, histtype='step', cumulative=True)
plt.plot([bins[n>.5][0],]*2, np.array((0,1.1)), color='red')
plt.ylim([0, 1.1])
plt.xlim([0,2**0.5],)
plt.title('Assume independence of TI and APC measurements in V4.')
plt.xlabel('Distance from "perfect" V4: APC = 1, TI = 1.')
nice_axes(plt.gcf().axes, xticks=None, yticks=None, nxticks=5, nyticks=2)
plt.tight_layout()
plt.savefig(figure_folder + 'v4ness_ofV4_assuming_independence_cumu.eps')

dist = ((1- kts_c['cv_ti'])**2 + (1- kts_c['apc'])**2)**0.5
plt.figure(figsize=(7,12))
stacked_hist_layers((dist).dropna(), logx=False, logy=False,
                    maxlim=False, xlim=[0, 2**0.5], bins=100, cumulative=False, normed=True)
plt.xlabel('Distance from "perfect" V4: APC = 1, TI = 1.')
plt.savefig(figure_folder + 'v4ness_of_caffenet.eps')

plt.figure(figsize=(7,12))
stacked_hist_layers((dist).dropna(), logx=False, logy=False, xlim=[0, 2**0.5],
                    maxlim=False, bins=100, cumulative=True, normed=True)
plt.xlim(0,2**0.5)
plt.xlabel('Distance from "perfect" V4: APC = 1, TI = 1.')
plt.savefig(figure_folder + 'v4ness_of_caffenet_cumu.eps')



for name in names:
    pda = pan[name]
    plt.figure()
    stacked_hist_layers((pda[pda['k']<k_thresh])['cv_ti'].dropna(), logx=False, logy=False, xlim=[0,1], maxlim=False, bins=100)
    plt.suptitle('3-fold Cross-Validated TI. Stimuli width: ' + name )
    plt.savefig('3-fold Cross-Validated TI. Stimuli width: ' + name + '.eps')
