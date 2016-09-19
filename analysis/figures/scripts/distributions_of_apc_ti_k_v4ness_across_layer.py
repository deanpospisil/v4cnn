# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:41:01 2016

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
                        extra_subplot=False, title=None):
    layers = cnn.index.get_level_values('layer_label').unique()
    if logx:
        cnn = np.log10(cnn.dropna())
        xlim = np.log10(xlim)
    if maxlim:
        xlim = [np.min(cnn.dropna().values), np.max(cnn.dropna().values)]

    n_subplot = len(layers)+extra_subplot
    for i, layer in enumerate(layers):
        plt.subplot(n_subplot, 1, i+1)

        if title!=None and i==0:
            plt.title(title)
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

figure_folder = top_dir + 'analysis/figures/images/'
cnn_names =['APC362_deploy_fixing_relu_saved.prototxt_fixed_even_pix_width[24.0, 48.0]_pos_(64.0, 164.0, 51)bvlc_reference_caffenet',
'APC362_deploy_fixing_relu_saved.prototxt_shuffle_fixed_even_pix_width[24, 30.0]_pos_(64.0, 164.0, 51)bvlc_caffenet_reference_shuffle']




da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[1] + '.nc')['resp'].isel(scale=0)
da = da - da[0]
rf = (da**2).sum('shapes')
rf_null = rf / rf.max('x')

da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=0)
da = da - da[0]
rf = (da**2).sum('shapes')
rf_alt = rf / rf.max('x')

plt.figure()
plt.subplot(211)
rf_null.plot()
plt.xticks(type_change, type_label, rotation='vertical', size = 'small')
plt.subplot(212)
rf_alt.plot()
plt.xticks(type_change, type_label, rotation='vertical', size = 'small')
plt.tight_layout()

indexes = np.unique(da_0.coords['layer_label'].values, return_index=True)[1]
layer_label = [da_0.coords['layer_label'].values[index] for index in sorted(indexes)]


if 'dmod' not in locals():
    fn = top_dir + 'data/models/' + 'apc_models_362.nc'
    dmod = xr.open_dataset(fn, chunks={'models': 50, 'shapes': 370}  )['resp'].load()


k_thresh = 40
names = ['24', '30']
name = names[1]

v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()
v4 = process_V4(v4_resp_apc, v4_resp_ti, dmod)

#shuffle
v4_resp_apc_null = v4_resp_apc.copy()
v4_resp_ti_null = v4_resp_ti.copy()

for  x in range(len(v4_resp_ti_null.coords['x'])):
    for unit in range(len(v4_resp_ti_null.coords['unit'])):
        not_null = ~v4_resp_ti_null[unit,x,:].isnull()
        v4_resp_ti_null[unit,x, not_null] = np.random.permutation(v4_resp_ti[unit,x,not_null].values)

for unit in range(len(v4_resp_apc_null.coords['unit'])):
    v4_resp_apc_null[:,unit] = np.random.permutation(v4_resp_apc[:,unit].values)

v4_null = process_V4(v4_resp_apc_null, v4_resp_ti_null, dmod)



with open(top_dir + 'data/an_results/fixed_relu_saved_24_30_pix.p', 'rb') as f:
    pan = pickle.load(f)
pda = pan[name]

with open(top_dir + 'data/an_results/null_fixed_relu_saved_24_48_pix.p', 'rb') as f:
    pan_null = pickle.load(f)

name = names[0]
pda_null = pan_null[name]

#%%


plt.close('all')
df = pan[name]
df = df[df.index.get_level_values('layer_label')!='prob']
all_layers = pd.concat([df, v4])
all_layers['apc'] = all_layers['apc']**2 #square here to get frac var for apc

df_null = pan_null[name]
df_null = df_null[df_null.index.get_level_values('layer_label')!='prob']
all_layers_null = pd.concat([df_null, v4_null])
all_layers_null['apc'] = all_layers_null['apc']**2


plt.scatter(df['cv_ti'][df['k']<40],df['ti'][df['k']<40], s=0.1, alpha =0.5)
plt.title('3-fold CV SVD vs SVD' + name +' pixel')
plt.plot([0,1],[0,1]);plt.xlim((0,1));plt.ylim((0,1))
plt.xlabel('3-fold CV TI');plt.ylabel('TI')
plt.savefig(figure_folder + name +'cv_vs_non_cv_ti.png')
plt.savefig(figure_folder + name +'cv_vs_non_cv_ti.eps')

plt.figure(figsize=(6,12))
title = '3-fold CV TI measured in receptive field.'+name +' pixel'
stacked_hist_layers((all_layers[all_layers['k']<k_thresh])['cv_ti'].dropna(),
                    title=title, logx=False, logy=True, xlim=[0,1], maxlim=False, bins=100)
stacked_hist_layers((all_layers_null[all_layers_null['k']<k_thresh])['cv_ti'].dropna(),
                    title=title, logx=False, logy=True, xlim=[0,1], maxlim=False, bins=100)
plt.xlabel('Fraction Variance Explained by TI Model')
plt.tight_layout();

plt.savefig(figure_folder +name + 'ti_frac_var_null_alt.eps')
plt.savefig(figure_folder + name +'ti_frac_var_null_alt.png')

plt.figure()
plt.subplot(121)
all_layers['cv_ti'].groupby(all_layers.index.get_level_values('layer_label')).median().loc[layer_label+['v4',]].plot(kind='bar')
plt.ylim(0,1)
plt.ylabel('Fraction Variance TI')
plt.title('unshuffled')
plt.subplot(122)
all_layers_null['cv_ti'].groupby(all_layers_null.index.get_level_values('layer_label')).median().loc[layer_label+['v4',]].plot(kind='bar')
plt.ylim(0,1)
plt.title('shuffled')

plt.savefig(figure_folder + '24_ti_median_nul_alt.png')


da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=0)
middle = np.round(len(da.coords['x'])/2.)
da_0 = da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)])
da_0_sc0 = da.sel(x=da.coords['x'][middle:middle+4]).squeeze()
da_0_sc0 = da_0_sc0 / da_0_sc0.chunk({}).vnorm('shapes').load()
da_0_sc0_cor = (da_0_sc0[:,0,:]*da_0_sc0[:,3,:]).sum('shapes')**2

da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=1)
da_0_sc1 = da.sel(x=da.coords['x'][middle:middle+4]).squeeze()
da_0_sc1 = da_0_sc1 / da_0_sc1.chunk({}).vnorm('shapes').load()
da_0_sc1_cor = (da_0_sc1[:,0,:]*da_0_sc1[:,3,:]).sum('shapes')**2

plt.figure()
bigger_better =((da_0_sc1_cor - da_0_sc0_cor)).groupby('layer_label').mean()
fraction_bigger_better = bigger_better/da_0_sc0_cor.groupby('layer_label').count().astype(float)

fraction_bigger_better.to_pandas()[layer_label].plot(kind='bar')
#plt.ylim(0,1)
#plt.plot([-1000,1000],[0.5,0.5])
plt.title('six pixel shift from center frac_var')
plt.ylabel('mean difference between frac_var(48)-frac_var(24) pixel')
plt.savefig(figure_folder +'24vs48_6pix_mean_frac_var_dif.png')


plt.figure()
units = pda.index.get_level_values('layer_label')=='fc8'
pda = pan['24']
plt.close('all')
plt.subplot(121)
plt.scatter(pda['ti_orf'][units][pda['k']<40], pda['ti'][units][pda['k']<40], s=0.1, alpha =0.5)
plt.xlabel('Outside RF');plt.ylabel('Inside RF')
plt.xlim(0,1);plt.ylim(0,1);plt.plot([0,1],[0,1])
plt.title('24 pixels FC8')

plt.subplot(122)
plt.scatter(pda['ti_orf'][pda['k']<40], pda['ti'][pda['k']<40], s=0.1, alpha =0.5)
plt.xlabel('Outside RF');plt.ylabel('Inside RF')
plt.xlim(0,1);plt.ylim(0,1);plt.plot([0,1],[0,1])
plt.title('24 pixels ALL LAYERS')
#plt.title('SVD over all pos vs SVD in rf' + name +' pixel')
plt.plot([0,1],[0,1]);plt.xlim((0,1));plt.ylim((0,1))
plt.savefig(figure_folder + name +'in_vs_outrf.png')
plt.savefig(figure_folder + name +'in_vs_outrf.eps')


from collections import Counter
pan['24']['in_rf'].groupby(pan['24'].index.get_level_values('layer_label')).apply(Counter)
pan['30']['in_rf'].groupby(pan['30'].index.get_level_values('layer_label')).apply(Counter)

#
#plt.figure()
#(all_layers['cv_ti']).groupby(all_layers.index.get_level_values('layer_label')).median()\
#[layer_label].plot(kind='bar')
#plt.title('ti_medians_over_layer')
#plt.ylabel('cv_ti ' +name)
#plt.ylim(0,1)
#plt.savefig(figure_folder + 'ti_medians_overlayer.png')
#
#da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=0)
#da = da.sel(unit=slice(0,None,1)).squeeze()
#da = da - da.mean('shapes')
#
#middle = np.round(len(da.coords['x'])/2.)
#da_0 = da.sel(x=da.coords['x'][np.round(len(da.coords['x'])/2.).astype(int)])
#non_zero_resp = da_0.var('shapes')!=0
#da = da[:,:,non_zero_resp]
#rf = da.chunk({}).vnorm('shapes').groupby('layer_label').mean('unit').load()
#rf = rf / rf.max('x')
#da_0 = da_0[:,non_zero_resp]
#
#da_0_nrm = da_0 / da_0.chunk({}).vnorm('shapes')
#da_nrm = da / da.chunk({}).vnorm('shapes')
#
#indexes = np.unique(da_0.coords['layer_label'].values, return_index=True)[1]
#layer_label = [da_0.coords['layer_label'].values[index] for index in sorted(indexes)]
#
#da_trans_cor = (da_0_nrm * da_nrm).sum('shapes').groupby('layer_label')\
#                .mean('unit').load().to_pandas()
#da_trans_cor = da_trans_cor.loc[layer_label]
#rf_p = rf.to_pandas().T.loc[layer_label]
#
##%%
#plt.close('all')
#plt.figure(figsize=(10,7))
#n_subplot=21
#for i, layer_val in enumerate(da_trans_cor.values):
#    plt.subplot(3,7, i+1, aspect=100)
#    #plt.subplot(n_subplot, 1, i+1)
#    plt.title(layer_label[i])
#    plt.plot(da.coords['x'].values, layer_val)
#    plt.plot(da.coords['x'].values, rf_p.values[i,:])
#    plt.ylim(-.5,1)
#    if i == 0:
#        plt.yticks([-.5,0,.5,1])
#    else:
#        plt.yticks([])
#    plt.xticks(da.coords['x'].values[::10])
#    plt.tick_params(axis='both', which='both', labelsize=5)
#    plt.grid()
#plt.suptitle('Correlation (blue) and Receptive field (green)')
#plt.tight_layout()
#plt.savefig(figure_folder +'24_correlation_rf.eps')
#plt.savefig(figure_folder + '24_correlation_rf.png')

#%%
#APC

with open(top_dir + 'data/an_results/fixed_relu_saved_24_30_pix.p', 'rb') as f:
    pan = pickle.load(f)


plt.figure(figsize=(6,12))
title = 'Center fit to APC model. ' + name + ' pix'
apc_fits_kts = all_layers[all_layers['k']<k_thresh]['apc'].dropna()
stacked_hist_layers(apc_fits_kts,
                    logx=False, logy=True, xlim=[0,1], maxlim=False, bins=100, title=title)
apc_fits_kts_null = all_layers_null[all_layers_null['k']<k_thresh]['apc'].dropna()
stacked_hist_layers(apc_fits_kts_null,
                    logx=False, logy=True, xlim=[0,1], maxlim=False, bins=100, title=title)
plt.xlabel('Fraction Variance Explained by APC Model')
plt.tight_layout()
plt.savefig(figure_folder + name +'apc_correlation_null_alt.eps')
plt.savefig(figure_folder +name + 'apc_correlation_null_alt.png')

plt.figure()
plt.subplot(121)
(apc_fits_kts).groupby(apc_fits_kts.index.get_level_values('layer_label')).\
    median()[layer_label+ ['v4',]].plot(kind='bar')

plt.xlabel('Median APC frac var')
plt.title(name+'pix unshuffled')
plt.ylim(0,1)
plt.grid()

plt.subplot(122)
(apc_fits_kts_null).groupby(apc_fits_kts_null.index.get_level_values('layer_label')).\
    median()[layer_label+ ['v4',]].plot(kind='bar')

plt.xlabel('Median APC frac var')
plt.title('shuffled')
plt.ylim(0,1)
plt.grid()


plt.savefig(figure_folder + name +'apc_correlation_median_null_alt.eps')
plt.savefig(figure_folder +name + 'apc_correlation_median_null_alt.png')

#%%

plt.figure()
(pan['30']['apc'] - pan['24']['apc']).groupby(pan['30'].\
index.get_level_values('layer_label')).median()[layer_label].plot(kind='bar')
plt.title('size and correlation APC difference')
plt.ylabel('median correlation difference 30-24')
plt.savefig(figure_folder + 'size_and_apc_median.png')


plt.figure(figsize=(6,12))
title = 'Difference in APC fit 30-24'
stacked_hist_layers((pan['30']['apc'] - pan['24']['apc']).dropna(), logx=False, logy=False, xlim=[-1, 1],
                        maxlim=False, bins=100, cumulative=False, normed=True,
                        extra_subplot=False, title=title)
plt.savefig(figure_folder + 'distribution_of_apc_size_cor_dif.png')


plt.figure()
plt.subplot(121)
all_layers['apc'].groupby(all_layers.index.get_level_values('layer_label')).median().loc[layer_label].plot(kind='bar')
plt.ylim(0,1)
plt.ylabel('Fraction Variance TI')
plt.title('unshuffled')
plt.subplot(122)
all_layers_null['apc'].groupby(all_layers_null.index.get_level_values('layer_label')).median().loc[layer_label].plot(kind='bar')
plt.ylim(0,1)
plt.title('shuffled')
plt.savefig(figure_folder + '24_apc_median_nul_alt.png')



#%%
#sparsity
plt.figure(figsize=(6, 12))
title = 'Sparsity of Layers. ' + name + ' pixel'
stacked_hist_layers(all_layers['k'].dropna(), logx=True, logy=True,
                    xlim=[min(all_layers['k']), max(all_layers['k'])],
                    maxlim=False, bins=100, title=title)
plt.xlabel('Kurtosis')
plt.savefig(figure_folder + name +'sparsity_loghist.eps')
plt.savefig(figure_folder + name +'sparsity_loghist.png')

plt.close('all')
plt.subplot(211)
k_apc = kurtosis(v4_resp_apc).values
plt.hist(v4_resp_apc[:, np.argmax(k_apc)], bins=30, log=True, normed=False, histtype='step',  range=[0,1])
plt.hist(v4_resp_apc[:, np.argmin(k_apc)], bins=30,log=True, normed=False, histtype='step', range=[0,1])
plt.legend(np.round([np.min(k_apc), np.max(k_apc)],1), title='Kurtosis', loc=1)
plt.xlabel('Normalized firing rate')
plt.ylabel('Firing rate density')
plt.plot([0,1],[0,0])
plt.title('Response distribution of most and least sparse V4 unit.')

plt.subplot(212)
plt.hist(np.log10(pda['k'].dropna()),  bins=100, histtype='step', normed=True,
         range=np.log10([min(all_layers['k']), max(all_layers['k'])]), color='m')
plt.hist(np.log10(k_apc), bins=100, histtype='step',normed=True,
         range=np.log10([min(all_layers['k']), max(all_layers['k'])]), color='c')
plt.xlabel('Log10 Kurtosis')
plt.ylabel('Density')
plt.title('Comparison of Kurtosis Distributions')
plt.legend(['CaffeNet','V4'], loc=9)
plt.tight_layout()
#nice_axes(plt.gcf().axes, xticks=None, yticks=None, nxticks=5, nyticks=4)
plt.savefig(figure_folder + name +'sparsity_V4_vs_caffe.eps')
plt.savefig(figure_folder + name +'sparsity_V4_vs_caffe.png')

plt.figure(figsize=(6,12))
title = 'Difference in Sparsity 30-24 '
stacked_hist_layers((pan['30']['k'] - pan['24']['k']).dropna(), logx=False, logy=True, xlim=[-370,370],
                        maxlim=False, bins=100, cumulative=False, normed=True,
                        extra_subplot=False, title=title)
plt.savefig(figure_folder + 'distribution_of_k_size_cor_dif.png')


#%%
#v4ness
plt.close('all')
kts = all_layers[all_layers['k']<k_thresh]
kts_c = kts[kts.index.get_level_values('layer_label')!='v4']
kts_v = kts[kts.index.get_level_values('layer_label')=='v4']
ti_m_prob, ti_m_value = np.histogram(kts_v['cv_ti'].dropna(), normed=True, bins=20)
apc_m_prob, apc_m_value  = np.histogram(kts_v['apc'].dropna(), normed=True, bins=20)

joint_v4ness = apc_m_prob.reshape(1,20)*ti_m_prob.reshape(20,1)
apc_val, ti_val = np.meshgrid(apc_m_value[1:], ti_m_value[1:])
dist_v4 = ((1- apc_val)**2 + (1- ti_val)**2)**0.5
n, bins, patches = plt.hist(dist_v4.ravel(), weights=joint_v4ness.ravel(),
                            bins=100, range=[0,1], normed=True, histtype='step',
                            cumulative=True)
dist_c = ((1- kts_c['cv_ti'])**2 + (1- kts_c['apc'])**2)**0.5

plt.close('all')
plt.figure(figsize=(6,12))
stacked_hist_layers((dist_c).dropna(), logx=False, logy=False, xlim=[0, 2**0.5],
                    maxlim=False, bins=100, cumulative=True, normed=True, extra_subplot=True)

plt.subplot(22,1,22)
plt.gca().xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
plt.hist(dist_v4.ravel(), weights=joint_v4ness.ravel(),bins=50,
                            range=[0, 2**0.5], normed=True, histtype='step', cumulative=True)
v4_median_v4ness = bins[n>.5][0]
plt.plot([v4_median_v4ness,]*2, np.array((0,1.1)), color='red')
plt.ylim([0, 1.1])
plt.xlim([0,2**0.5],)
plt.gca().set_ylabel('V4', ha='right', rotation=0, labelpad=25)
plt.gca().yaxis.set_label_position("right")
#plt.title('Assume independence of TI and APC measurements in V4.' +name +' pixel')
#plt.xlabel('Distance from "perfect" V4: APC = 1, TI = 1.')
nice_axes(plt.gcf().axes, xticks=None, yticks=None, nxticks=5, nyticks=2)
plt.xlabel('Distance from "perfect" V4: APC = 1, TI = 1. ' +name +' pixel')

plt.savefig(figure_folder + name +'v4ness_ofV4_assuming_independence_and_caffe_cumu.eps')
plt.savefig(figure_folder + name +'v4ness_ofV4_assuming_independence_and_caffe_cumu.png')


plt.figure(figsize=(6,12))
stacked_hist_layers((dist_c).dropna(), logx=False, logy=False, xlim=[0, 2**0.5],
                    maxlim=False, bins=100, cumulative=False, normed=True,
                    extra_subplot=True)

plt.subplot(22,1,22)
plt.gca().xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
plt.hist(dist_v4.ravel(), weights=joint_v4ness.ravel(),bins=50,
                            range=[0, 2**0.5], normed=True, histtype='step', cumulative=False)
plt.plot([bins[n>.5][0],]*2, np.array((0,3.5)), color='red')
#plt.ylim([0, max()])
plt.xlim([0,2**0.5],)
plt.gca().set_ylabel('V4', ha='right', rotation=0, labelpad=25)
plt.gca().yaxis.set_label_position("right")
#plt.title('Assume independence of TI and APC measurements in V4.' +name +' pixel')
#plt.xlabel('Distance from "perfect" V4: APC = 1, TI = 1.')
nice_axes(plt.gcf().axes, xticks=None, yticks=None, nxticks=5, nyticks=2)
plt.xlabel('Distance from "perfect" V4: APC = 1, TI = 1.' +name +' pixel')
plt.savefig(figure_folder + name +'v4ness_ofV4_assuming_independence_and_caffe.eps')
plt.savefig(figure_folder + name +'v4ness_ofV4_assuming_independence_and_caffe.png')


plt.figure(figsize=(15,5))
s=4
sample_from_layers = ['fc8', 'conv5', 'norm2']
colors = ['r' , 'g' , 'b']
plt.subplot(131)
for c, layer in zip(colors, sample_from_layers):
    plt.scatter(kts_c.loc[layer]['cv_ti'][:300], kts_c.loc[layer]['apc'][:300],
                s=s, color=c, alpha=0.5)
    plt.xlabel('Fraction Variance TI');plt.ylabel('Fraction Variance APC')
    plt.xlim(0,1);plt.ylim(0,1)
plt.legend(sample_from_layers)
plt.title('300 random samples ')
plt.subplot(132)
plt.title('all layers')
plt.scatter(kts_c['cv_ti'], kts_c['apc'], s=0.1, alpha=0.5)

plt.xlabel('Fraction Variance CV TI');plt.ylabel('Fraction Variance APC')
plt.xlim(0,1);plt.ylim(0,1)
print(dist_c.argmin())
print(dist_c.min())
print(dist_c.sort(inplace=False))
plt.subplot(133)
med_v4ness = pd.concat([dist_c.groupby(dist_c.index.get_level_values('layer_label'))
                        .median()[layer_label], pd.Series({'v4':v4_median_v4ness})])
med_v4ness.plot(kind='bar')

plt.ylim(0.5, 1)
plt.xlabel('Median V4ness distance')
plt.tight_layout()
plt.savefig(figure_folder + name + 'ti_apc_plane_Samples.png')
