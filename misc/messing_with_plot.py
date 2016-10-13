# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:54:18 2016

@author: deanpospisil
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt 
import numpy as np
from itertools import product
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import matplotlib
from matplotlib.ticker import FuncFormatter
import pickle as pk
import xarray as xr;import pandas as pd
import apc_model_fit as ac
import matplotlib.ticker as mtick;
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp


def naked_plot(axes):
    for ax in  axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
def fit_axis(ax, x, y, x_ax=True, y_ax=True, unity_ax=False):
    maxest = max([max(x), max(y)])
    minest = min([min(x), min(y)])
    if unity_ax:
        ax.plot([minest,maxest],[minest,maxest], lw=0.1, color='k');
    if min(y)<0:
        ax.plot([min(x),max(x)],[0,0], lw=.3, color='k');
    if min(x)<0:
        ax.plot([0,0],[min(y),max(y)], lw=.3, color='k');

def simple_hist_pd(ax, var, orientation='vertical', fontsize=10):
    n = ax.hist(var.values, histtype='step', align='mid',lw=0.5, 
                bins='auto', orientation=orientation)[0]
    sigfig = 2
    naked_plot([ax,])
    the_range = [min(var), max(var)]
    if orientation[0]=='v':  
        ax.set_ylim(0, max(n)+max(n)*.15)
        ax.set_xlim(the_range)
        ax.text(np.max(var), -ax.get_ylim()[1]/10, np.round(np.max(var),sigfig),ha='right',va='top', fontsize=fontsize )
        ax.text(np.min(var), -ax.get_ylim()[1]/10, np.round(np.min(var),sigfig),  ha='left',va='top', fontsize=fontsize)
        spine_loc = 'bottom'
    else:
        ax.set_xlim(0, max(n)+max(n)*.15)
        ax.set_ylim(the_range)
        ax.text(-ax.get_xlim()[1]/10, np.max(var), np.round(np.max(var), sigfig),  ha='right',va='top', fontsize=fontsize )
        ax.text(-ax.get_xlim()[1]/10, np.min(var), np.round(np.min(var), sigfig),  ha='right',va='bottom', fontsize=fontsize)
        spine_loc = 'left'
    ax.spines[spine_loc].set_visible(True)
    ax.spines[spine_loc].set(lw=0.5)
    ax.spines[spine_loc].set_bounds(the_range[0], the_range[1])
    
# number of cols is the number of y variables, and number of rows
def small_mult_hist(x, labels, scale=1):
    m = max(list(map(len, cnn_val_lists)))
    gs = gridspec.GridSpec(m, 1, width_ratios=[1,],
                            height_ratios=[1,]*m)
    plt.figure(figsize=(4*scale, m*2*scale))
    
    fontsize = 10
    y_hists = []
    n_list = []
    sigfig = 1
    max_list = np.zeros((m, len(x)))
    min_list = np.zeros((m, len(x)))
    for i_an_x, an_x in enumerate(x):
        for x_col, pos in zip(an_x, range(m)):
            var = np.array(x_col)
            max_list[pos, i_an_x] = np.max(var)
            min_list[pos, i_an_x] = np.min(var)
            
    max_list = np.max(max_list, 1)
    min_list = np.min(min_list, 1)

    for an_x in x:
        for x_col, pos in zip(an_x, range(m)):
            ax = plt.subplot(gs[pos])
            var = np.array(x_col)
            the_range = (min_list[pos], max_list[pos])
            n, bins = np.histogram(var, bins=100, normed=False) 
            n =  n/float(len(var)-1);n = [0,] + list(n) + [0,];
            bins = [bins[0], ] + list(bins)
            ax.step(bins, n, where='mid', lw=0.5)
            ax.semilogy(nonposy='clip')
            ax.set_ylim(0.5/float(len(var)-1), np.max(n))
            naked_plot([ax,])
            
            ax.set_xlim(-np.max(np.abs(the_range)), np.max(np.abs(the_range)))
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set(lw=0.5)
            ax.spines['bottom'].set_bounds(-max(np.abs(the_range)), max(np.abs(the_range)))
            ax.set_xticks([the_range[0], 0, the_range[1]])
            ax.set_xticklabels([np.round(the_range[0],sigfig), ' ',
                                np.round(the_range[1],sigfig)])
            ax.set_ylabel(str(labels[pos]), rotation='horizontal', 
                         labelpad=fontsize*3, fontsize=fontsize)
            ax.yaxis.set_label_position('right')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.set_yticks([np.max(n),1])
            ax.set_yticklabels([np.round(np.max(n), 2),1])
            #ax.set_yticklabels([])
            
            y_hists.append(ax) 
            n_list.append(n)
#        ax.set_xticklabels([np.round(the_range[0],sigfig), 0,
#                            np.round(the_range[1],sigfig)])
    return y_hists, n_list
def small_mult_scatter_w_marg_pd(x, y):
    m = y.shape[1]+1
    n = x.shape[1]+1
    left_bottom = m*n-n
    y_hist_pos = list(range(0, m*n, n))[:-1]
    x_hist_pos = list(range(left_bottom+1, m*n))

    
    scatter_inds = np.sort(list(set(range(m*n)) - (set(x_hist_pos) | set(y_hist_pos) | set([left_bottom,]))))
    cart_inds = list(product(range(m-1), range(n-1)))
    
    gs = gridspec.GridSpec(m, n, width_ratios=[1,]+[8,]*(n-1),
                            height_ratios=[8,]*(m-1)+[1,])
    
    plt.figure(figsize=(n*2,m*2))
    fontsize=10
    y_hists = []
    for y_col, pos in zip(y, y_hist_pos):
        _=plt.subplot(gs[pos])
        print(y_col)
        simple_hist_pd(_, y[y_col], orientation='horizontal')
        _.set_ylabel(str(y_col), rotation='horizontal', 
                     labelpad=fontsize*3, fontsize=fontsize)
        y_hists.append(_)
    x_hists = []
    
    for x_col, pos in zip(x, x_hist_pos):
        _ = plt.subplot(gs[pos])
        simple_hist_pd(_, x[x_col])
        _.set_xlabel(str(x_col), rotation='horizontal', 
                      fontsize=fontsize, labelpad=fontsize*2)
        x_hists.append(_)
    
    scatters = []    
    for (y_ind, x_ind), pos in zip(cart_inds, scatter_inds):
        _ = plt.subplot(gs[pos], sharex= x_hists[x_ind], sharey=y_hists[y_ind])
        _.scatter(x.iloc[:, x_ind], y.iloc[:, y_ind], s=0.4)
        fit_axis(_, x.iloc[:, x_ind], y.iloc[:, y_ind])
        scatters.append(_)
          
    naked_plot(scatters)
    
    return scatters, x_hists, y_hists
'''
goforit = False
if 'fit_best_mods_pd' not in locals() or goforit:
    v4_name = 'V4_362PC2001'
    v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    fn = top_dir + 'data/models/' + 'apc_models_362.nc'
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()
    
    apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                      dmod.chunk({}), 
                                      fit_over_dims=None, 
                                      prov_commit=False)
    v4_resp_apc = v4_resp_apc - v4_resp_apc.mean('shapes')
    v4_resp_apc_pd = v4_resp_apc[:,apc_fit_v4.argsort().values].to_pandas()
    
    best_mods_pd = dmod[:, apc_fit_v4[apc_fit_v4.argsort().values]
                      .squeeze().coords['unit'].models.values]
    
    
    fn = top_dir + 'data/models/' + 'apc_models_362.nc'
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()
    apc_fit_v4 = apc_fit_v4**2
    fit_best_mods_pd = []
    for mod, resp in zip(best_mods_pd.values.T, v4_resp_apc_pd.values.T):
        mod = np.expand_dims(mod, 1)
        resp = np.expand_dims(resp, 1)
        fit_best_mods_pd.append(np.dot(mod, np.linalg.lstsq(mod, resp)[0]))
    fit_best_mods_pd = np.array(fit_best_mods_pd).squeeze().T
    fit_best_mods_pd = pd.DataFrame(fit_best_mods_pd)
                                    #columns=np.round(np.sort(apc_fit_v4.values),3))


names = [ 'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51).nc',
'bvlc_reference_caffenetAPC362_pix_width[64.0]_pos_(64.0, 164.0, 51).nc',
]
names = [ 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51).nc',
'bvlc_reference_caffenetAPC362_pix_width[64.0]_pos_(64.0, 164.0, 51).nc',
]

cnn_val_lists = []
for name in names:
    cnn = xr.open_dataset(top_dir + 'data/responses/' + name)['resp']
    cnn = cnn.sel(x=114).squeeze()
    cnn = cnn.transpose('unit', 'shapes')

    all_lays = cnn.coords['unit'].layer_label.values.astype(str)
    unique_inds = np.unique(all_lays, return_index=True)[1]
    layers = [all_lays[ind] for ind in np.sort(unique_inds)]
    type_change = list(np.where(np.diff(cnn.coords['layer'].values))[0])
    cnn = xr.open_dataset(top_dir + 'data/responses/' + name)['resp']
    cnn = cnn.sel(x=114).squeeze()
    cnn = cnn.transpose('unit', 'shapes')
    
    cnn_val_lists.append([cnn[a_layer==all_lays,].values.flatten() for a_layer in layers])
    
hists , n_list = small_mult_hist(cnn_val_lists, layers)
hists[0].legend(['32', '64'], frameon=0)
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/' + '32_64_pix_response_dist.eps')

name = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(114.0, 114.0, 1)_amp_(100, 255, 2).nc'
cnn = xr.open_dataset(top_dir + 'data/responses/' + name)['resp']
cnn = cnn.squeeze()
cnns = cnn.transpose('amp', 'unit', 'shapes')
cnn_val_lists = []
for cnn in cnns:
    all_lays = cnn.coords['unit'].layer_label.values.astype(str)
    unique_inds = np.unique(all_lays, return_index=True)[1]
    layers = [all_lays[ind] for ind in np.sort(unique_inds)]
    type_change = list(np.where(np.diff(cnn.coords['layer'].values))[0])
    cnn_val_lists.append([cnn[a_layer==all_lays,].values.flatten() for a_layer in layers])
    
hists , n_list = small_mult_hist(cnn_val_lists, layers)
hists[0].legend(['100', '255'], frameon=0, title='')
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/' + '100_255_amp_response_dist.eps')
'''

names = [ 'bvlc_reference_caffenetAPC362_pix_width[64.0]_pos_(64.0, 164.0, 51).nc',
'bvlc_reference_caffenet_nat_image_resp_371.nc',
]
cnn = xr.open_dataset(top_dir + 'data/responses/' + names[0])['resp']
cnn = cnn.sel(x=114).squeeze()
cnn = cnn.transpose('unit', 'shapes')
cnn1 = cnn

cnn = xr.open_dataset(top_dir + 'data/responses/' + names[1])['resp']
cnn2 = cnn
cnns = [cnn1,cnn2]
cnn_val_lists = []
for cnn in cnns:
    cnn = cnn.transpose('unit', 'shapes')

    all_lays = cnn.coords['unit'].layer_label.values.astype(str)
    unique_inds = np.unique(all_lays, return_index=True)[1]
    layers = [all_lays[ind] for ind in np.sort(unique_inds)]
    type_change = list(np.where(np.diff(cnn.coords['layer'].values))[0])
    cnn_val_lists.append([cnn[a_layer==all_lays,].values.flatten() for a_layer in layers])

hists , n_list = small_mult_hist(cnn_val_lists, layers)
hists[0].legend(['APC', 'ImageNet'], frameon=0, title='Image Type', fontsize='small')
hists[0].set_xlabel('Response')
hists[0].annotate('% (log-axis)', xy=(-0.1, 0.4), xycoords='axes fraction', 
                    rotation='vertical', ha='center',va='bottom', fontsize='small')
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/' + 'art_nat_amp_response_dist.eps')
