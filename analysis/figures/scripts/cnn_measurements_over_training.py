# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:51:48 2016

@author: dean

#making plots over stages of training. need to get performanceormance data.
"""
import os, sys
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'common')

import pandas as pd
import d_misc as dm
import xarray as xr
import apc_model_fit as ac
all_iter = dm.list_files(top_dir + 'data/an_results/r_apc_models_u*.nc')

def tick_format_d(x, pos):
    if x==0:
        return('0')
    else:
        if x>=1:
            return(str(x).split('.')[0])
        else:
            return(np.round(x,2))

def nice_axes(axes, xticks=None, yticks=None, nxticks=5, nyticks=2):
    for i, an_axes in enumerate(axes):

        if yticks==None:
            an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
        else:
            an_axes.set_yticks(yticks)
        if xticks==None:
            an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
        else:
            an_axes.set_xticks(xticks)
        an_axes.xaxis.set_tick_params(length=0)
        an_axes.yaxis.set_tick_params(length=0)
        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))

f = open(top_dir + '/data/image_net/imagenet_log_May_21.txt', 'r')
log = f.readlines()

iter_loss = [re.findall('Iteration \d+.+loss = \d+.\d+', line) for line in log
            if not re.findall('Iteration \d+.+loss = \d+.\d+', line)==[]]
loss = np.array([np.double(re.split(' ',line[0])[-1]) for line in iter_loss])[:-1]
iteration = np.array([np.double(re.split(' ', re.split(', ',line[0])[0])[1] ) for line in iter_loss])[:-1]

lr = [re.findall(' lr = \d+.\d+', line) for line in log]

lr = np.array([float(re.split(' = ', line[0])[1]) for line in
                [re.findall(' lr = \d+.\d+| lr = \de-\d+', line) for line in log]
                if not line==[]])

acc = np.array([np.double(re.split(' = ', line[0])[1]) for line in
                [re.findall('accuracy = \d+.\d+', line) for line in log]
                if not line==[]])
acc_iter = np.array([int(re.split(',', line[0])[0]) for line in
                [re.findall('\d+, Testing net', line) for line in log]
                if not line==[]])

performance = pd.DataFrame([acc_iter, acc]).T
performance = performance.set_index(0)

all_iter = [name for name in dm.list_files(top_dir + 'data/an_results/sparsity_APC362_scale_0.45_pos_(-7, 7, 15)*.nc') if not 'ref' in name]
iter_numbers = [int(re.findall('\d+', line)[-1]) for line in all_iter]
all_iter = [all_iter[sort_i] for sort_i in np.argsort(iter_numbers)]
spar_iter = [xr.open_dataset(name)['spar'] for name in all_iter]
spar_all = xr.concat(spar_iter, xr.DataArray(np.sort(iter_numbers), dims='iter', name='iter'))

all_iter =[name for name in  dm.list_files(top_dir + 'data/an_results/ti_APC362_scale_0.45_pos_(-7, 7, 15)*.nc') if not 'ref' in name]
iter_numbers = [int(re.findall('\d+', line)[-1]) for line in all_iter]
all_iter = [all_iter[sort_i] for sort_i in np.argsort(iter_numbers)]
ti_iter = [xr.open_dataset(name)['tin'] for name in all_iter]
ti_all = xr.concat(ti_iter, xr.DataArray(np.sort(iter_numbers), dims='iter', name='iter'))

all_iter = [name for name in dm.list_files(top_dir + 'data/an_results/apc_APC362_scale_0.45_pos_(-7, 7, 15)*.nc') if not 'ref' in name]
iter_numbers = [int(re.findall('\d+', line)[-1]) for line in all_iter]
all_iter = [all_iter[sort_i] for sort_i in np.argsort(iter_numbers)]
apc_iter = [xr.open_dataset(name)['r'].reindex_like(spar_iter[0]) for name in all_iter]
apc_all = xr.concat(apc_iter, xr.DataArray(np.sort(iter_numbers), dims='iter', name='iter'))


apc_all = apc_all.reindex_like(spar_all)
ti_all = ti_all.reindex_like(spar_all)

dv = [ti_all, apc_all, spar_all]
iter_numbers=np.sort(iter_numbers)

index_names = ['layer_label', 'layer', 'layer_unit',  'unit']
all_vals = []
for iteration in spar_all.coords['iter'].values:
    temp_vals = np.vstack([d.sel(iter=iteration).values for d in dv])
    temp_vals = np.vstack([np.tile(performance.loc[iteration], temp_vals.shape[1]), temp_vals])
    #its picking up your reference network 0
    temp_ind = np.vstack([dv[0].coords[name].values for name in index_names])
    temp_ind = np.vstack([np.tile(iteration, temp_ind.shape[1]), temp_ind])

    temp_vals = np.vstack([ temp_ind, temp_vals])
    all_vals.append(temp_vals)

all_vals = np.hstack(all_vals).T
df = pd.DataFrame(all_vals,
              columns=['iteration', 'layer_label', 'layer', 'layer_unit',  'unit', 'performance', 'ti', 'apc', 'spar'])
df = df.set_index(['iteration','layer_unit', 'layer_label', 'unit', 'layer'], drop=True).astype(np.double)

apc_thresh = 0.5
ti_thresh = 0.1
spar_thresh = 0.8

v4like = ((df.spar<spar_thresh)*(df.ti>ti_thresh)*(df.apc>apc_thresh))

import collections
counts_in_layers = collections.Counter(v4like.loc['0'].reset_index()['layer_label'].values)
v4like = v4like.groupby(level=['iteration', 'layer_label']).sum()

_ = pd.concat([v4like.loc[iteration] for iteration
                in v4like.index.levels[0].values],
                axis=1)
_.columns = v4like.index.levels[0].values.astype(np.int)
v4like_over_training = _.sort_index(axis=1)
v4like_over_training.T.plot()
plt.scatter(df.loc['0'].apc.values, df.loc['0'].ti.values, color='red')
plt.scatter(df.loc['450000'].apc.values, df.loc['450000'].ti.values)

plt.scatter(df.loc['450000'].apc.values, df.loc['450000'].ti.values)

axes = pd.tools.plotting.scatter_matrix(df.loc['0'], alpha=0.2)
plt.tight_layout()