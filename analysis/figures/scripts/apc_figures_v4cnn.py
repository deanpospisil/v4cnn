#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:14:18 2016

@author: dean
"""

import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr;import pandas as pd
import apc_model_fit as ac
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp
import d_net_analysis as dn
from sklearn.neighbors import KernelDensity
import caffe_net_response as cf

def beautify(ax=None, spines_to_remove = ['top', 'right']):
    almost_black = '#262626'
    more_grey = '#929292'
#    text_font = 'serif'
#    number_font = 'helvetica'
    all_spines = ['bottom','left','right','top']
    # Get the axes.
    if ax is None:
        #fig = plt.figure(1)
        ax = plt.axes()
    if not type(ax)==type([]):
        ax = [ax,]
    for a_ax in ax:
        # Remove 'spines' (axis lines)
        for spine in spines_to_remove:
            a_ax.spines[spine].set_visible(False)
    
        # Make ticks only where there are spines
        if 'left' in spines_to_remove:
            a_ax.tick_params(left=0)
        if 'right'  in spines_to_remove:
            a_ax.tick_params(right=0)
        if 'bottom'  in spines_to_remove:
            a_ax.tick_params(bottom=0)
        if 'top'  in spines_to_remove:
            a_ax.tick_params(top=0)
    
        # Now make them go 'out' rather than 'in'
        for axis in ['x', 'y']:
            a_ax.tick_params(axis=axis, which='both', direction='out', pad=7)
            a_ax.tick_params(axis=axis, which='major', color=almost_black, length=6)
            a_ax.tick_params(axis=axis, which='minor', color=more_grey, length=4)
    
        # Make thinner and off-black
        spines_to_keep = list(set(all_spines) - set(spines_to_remove))
        
        for spine in spines_to_keep:
            a_ax.spines[spine].set_linewidth(0.5)
            a_ax.spines[spine].set_color(almost_black)

    
        # Change the labels & title to the off-black and change their font
        for label in [a_ax.yaxis.label, a_ax.xaxis.label, a_ax.title]:
            label.set_color(almost_black)
    
        # Change the tick labels' color and font and padding
        for axis in [a_ax.yaxis, a_ax.xaxis]:
            # padding
            axis.labelpad = 20
            # major ticks
            for major_tick in axis.get_major_ticks():
                label = major_tick.label
                label.set_color(almost_black)
            # minor ticks
            for minor_tick in axis.get_minor_ticks():
                label = minor_tick.label
                label.set_color(more_grey)

    #plt.grid(axis='y', color=more_grey)

def scatter_lsq(ax, a, b, lsq=True, mean_subtract=True, **kw):    

    if len(a.shape)<=1:
        a = np.expand_dims(a,1)
    if len(b.shape)<=1:
        b = np.expand_dims(b,1)
    
    if mean_subtract:
        a -= np.mean(a);b -= np.mean(b)
    if a.shape[1] > 1 :
        print('a second dim to big, just taking the first col')
        a = a[:,1]
    if b.shape[1] > 1 :
        print('b second dim to big, just taking the first col')
        b = b[:,1]   
    if lsq:
        x = np.linalg.lstsq(a, b)[0]
        a_scaled = np.dot(a, x)
    else:
        a_scaled = a
    ax.scatter(a_scaled, b, **kw)
    return a_scaled, b
    
    
def cartesian_axes(ax, x_line=True, y_line=True, unity=False):
    more_grey = '#929292'
    ax.axis('equal')
    ylim = ax.spines['left'].get_bounds()
    xlim = ax.spines['bottom'].get_bounds() 
    if xlim == None:
        xlim = ax.get_xlim()
    if ylim == None:
        ylim = ax.get_ylim()
    
    if x_line:
        #ax.spines['bottom'].set_position('center')
        ax.plot([xlim[0], xlim[1]], [0, 0], color=more_grey,lw=0.5, alpha=0.5)
    if y_line:
        #ax.spines['left'].set_position('center')
        ax.plot([0, 0], [ylim[0], ylim[1]], color=more_grey,lw=0.5,alpha=0.5)
    
    if unity:
        ax.plot(xlim, xlim, color=more_grey, lw=0.5,alpha=0.5)

    
        
def data_spines_twinx(ax, x, y, mark_zero=[True, False], sigfig=2, fontsize=12, 
                nat_range=None, minor_ticks=False, data_spine=['bottom', 'left']):
    
    if not type(ax)==type([]):
        ax = [ax,]
    if not type(y)==type([]):
        y = [y,]
    if (not type(nat_range)==type([])) and (not nat_range == None):
        nat_range = [nat_range,]
    if nat_range == None:
            nat_range = [[[np.min(x), np.max(x)], [np.min(y[0]), np.max(y[0])]],
                         [[np.min(x), np.max(x)], [np.min(y[1]), np.max(y[1])]]]

    for i, a_ax in enumerate(ax):
        #todo minor ticks, nat_range
        ticks = [[np.min(x), np.max(x)],[np.min(y[i]), np.max(y[i])]]

        a_ax.spines['bottom'].set_bounds(nat_range[0][0][0], nat_range[0][0][1])
        
        if i==0:
            a_ax.spines['left'].set_bounds(nat_range[0][1][0], nat_range[0][1][1])
            
            scale = np.diff(ax[1].get_ylim())/np.diff(ax[0].get_ylim())
            scale= scale**-1
            a_ax.spines['right'].set_bounds(nat_range[1][1][0]*scale, nat_range[1][1][1]*scale)
        else:
            a_ax.spines['right'].set_bounds(nat_range[1][1][0], nat_range[1][1][1])
            
            scale = np.diff(ax[0].get_ylim())/np.diff(ax[1].get_ylim())
            scale= scale**-1
            
            a_ax.spines['left'].set_bounds(nat_range[0][1][0]*scale, nat_range[0][1][1]*scale)
        
        for axis in range(2):
            if abs((0-ticks[axis][0])/(ticks[axis][0] - ticks[axis][1]))<.05:
                mark_zero[axis] = False
            if mark_zero[axis]:
                ticks[axis] += [0,]
    
        a_ax.set_xticks(ticks[0])
        a_ax.set_xticklabels(np.round(ticks[0], sigfig), fontsize=fontsize)
        a_ax.set_yticks(ticks[1])
        a_ax.set_yticklabels(np.round(ticks[1], sigfig), fontsize=fontsize)
        a_ax.set_ylim(bottom=a_ax.get_ylim()[0] +a_ax.get_ylim()[0]*0.1, 
                      top=a_ax.get_ylim()[1]+a_ax.get_ylim()[1]*0.1)
    
    return None
def data_spines(ax, x, y, mark_zero=[True, False], sigfig=2, fontsize=12, 
                nat_range=None, minor_ticks=False, 
                data_spine=['bottom', 'left'],
                 supp_xticks=None, supp_yticks=None):

    if nat_range == None:
            nat_range = [[np.min(x), np.max(x)], [np.min(y), np.max(y)]]
    
    #todo minor ticks, nat_range
    ticks = [[np.min(x), np.max(x)],[np.min(y), np.max(y)]]
    ax.spines['bottom'].set_bounds(nat_range[0][0], nat_range[0][1])
    ax.spines['left'].set_bounds(nat_range[1][0], nat_range[1][1])
   
    
    for axis in range(2):
        if abs((0-ticks[axis][0])/(ticks[axis][0] - ticks[axis][1]))<.05:
            mark_zero[axis] = False
        if mark_zero[axis]:
            ticks[axis] += [0,]

    if supp_xticks != None:
        ticks[0] += supp_xticks
    if supp_yticks != None:
        ticks[1] += supp_yticks
    ax.set_xticks(ticks[0])
    ax.set_xticklabels(np.round(ticks[0], sigfig), fontsize=fontsize)
    ax.set_yticks(ticks[1])
    ax.set_yticklabels(np.round(ticks[1], sigfig), fontsize=fontsize)
    
    return None
def d_cust_hist(ax, n, bins, color='k'):
    for a_n, a_bin in zip(n, bins):
        ax.plot([a_bin,a_bin],[0, a_n], color=color, lw=0.9, linestyle=':', alpha=0.7)
    for i, a_n in enumerate(n):
        ax.plot([bins[i], bins[i+1]], [a_n, a_n],color=color)     
    
def d_hist(ax, x, bins='auto', alpha=0.5, color='k', normed=True, cumulative=False):
    if bins=='auto':
        bins = np.round(np.sqrt(len(x))/2)
    if cumulative:
        y_cum = np.array(range(1,len(x)+1))/float(len(x))
        ax.step(np.sort(x), y_cum, 
                alpha=alpha, lw=1, color=color)
        ax.scatter([np.max(x)], [1,], color=color, marker='|')
        n = y_cum
        bins = np.sort(x)
    else:
        n, bins = np.histogram(x, bins=bins, normed=True)
        d_cust_hist(ax, n, bins=bins, color=color)
        #n, bins, _ = ax.hist(x, bins=bins, color=color, histtype='step', 
        #                 alpha=alpha, lw=1, normed=normed)
        
    return n, bins
                 
def kde_dist(ax, x, bw=None, color='k'):
    x_grid = np.linspace(np.min(x), np.max(x), 1000)
    if bw == None:
        bw = np.std(x)*float(len(x))**(-1/5.)
    kde_skl = KernelDensity(bandwidth=bw)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    est = np.exp(log_pdf)
    ax.plot(x_grid, est, color=color, lw=0.5)
    
    return est
def vis_square(ax, data, padsize=0, padval=0):
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    ax.set_xticks([]);ax.set_yticks([])
#    if min((data.ravel())>=0):
#        clim = (min(abs(data.ravel())), max(abs(data.ravel())))
#    else:
#        clim = (-max(abs(data.ravel())), max(abs(data.ravel())))

    im = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
    #cbar=ax.colorbar(shrink=0.8)
    #cbar.ax.set_ylabel('Response', rotation= 270, labelpad=15, fontsize = 15,)
#    cbar.ax.yaxis.set_ticks([0,.25,.5,.75, 1])
#    cbar.ax.set_yticklabels(['0', .25, .5, .75, 1])
    #cbar.solids.set_rasterized(True)
    return im
    
def plot_resp_on_shapes(ax, imgStack, resp, image_square = 19):
    resp_sc = (resp.values)
    imgStack = imgStack*resp_sc.reshape(362,1,1)
    #sort images
    sortStack = imgStack[list(reversed(np.argsort(resp_sc))),:,:]
    sortStack = np.array([imp.centeredCrop(img, 64, 64) for img in sortStack])
    im = vis_square(ax, sortStack[0:image_square**2])
    return im

def open_cnn_analysis(fn,layer_label):
    try:
        an=pk.load(open(top_dir + 'data/an_results/' + fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(top_dir + 'data/an_results/' + fn,'rb'))
    fvx = an[0].sel(concat_dim='r2')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    return fvx, rf, cnn

def process_V4(v4_resp_apc, v4_resp_ti, dmod):
    ti = dn.ti_av_cov(v4_resp_ti, rf=None)
    apc = dn.ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                  dmod.chunk({}), fit_over_dims=None, 
                                    prov_commit=False)**2.
    k_apc = list(dn.kurtosis(v4_resp_apc).values)
    k_ti = list(dn.kurtosis(v4_resp_ti.mean('x')).values)

    keys = ['layer_label', 'unit']
    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(ti)),
                                       np.arange(len(ti))], names=keys)
    v4pdti  = pd.DataFrame(np.array([ti, k_ti]).T, index=index, 
                           columns=['ti_av_cov', 'k'])

    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(apc)), 
                                       np.arange(len(apc))], 
                                       names=keys)
    coords_to_take= ['cur_mean', 'cur_sd', 'or_mean', 'or_sd', 'models']
    apc_coords = [apc.coords[coord].values for coord in coords_to_take]
    v4pdapc  = pd.DataFrame(np.array([apc.values,] + apc_coords + [k_apc,]).T, 
               index=index, 
               columns=['apc', ] + coords_to_take + [ 'k',])
    v4 = pd.concat([v4pdti, v4pdapc])
    return v4
#%%
#shape image set up
img_n_pix = 227
max_pix_width = [64,]
s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(-1, 370); center_image = round(img_n_pix/2)
x = (center_image, center_image, 1);
y = (center_image, center_image, 1)
stim_trans_cart_dict, _ = cf.stim_trans_generator(shapes=shape_ids, scale=scale, 
                                                  x=x, y=y)
#plt.figure(figsize=(12,24));
center = 114
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict, 
                                                        base_stack, 
                                                        npixels=227))
no_blank_image = trans_img_stack[1:]
a = np.hstack((range(14), range(18, 318)));a = np.hstack((a, range(322, 370)))
no_blank_image = no_blank_image[a]/255.

goforit = True
#loading up all needed data
if 'cnn_an' not in locals() or goforit:
    v4_name = 'V4_362PC2001'
    v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    file = open(top_dir + 'data/responses/v4_apc_109_neural_labels.txt', 'r')
    wyeth_labels = [label.split(' ')[-1] for label in 
                file.read().split('\n') if len(label)>0]
    v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
    fn = top_dir + 'data/models/' + 'apc_models_362.nc'
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()
    
    apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                      dmod.chunk({}), 
                                      fit_over_dims=None, 
                                      prov_commit=False)**2

    v4_resp_apc = v4_resp_apc - v4_resp_apc.mean('shapes')
    v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()
    alt_v4 = process_V4(v4_resp_apc, v4_resp_ti, dmod)

    #shuffle
    v4_resp_apc_null = v4_resp_apc.copy()
    v4_resp_ti_null = v4_resp_ti.copy()

    for  x in range(len(v4_resp_ti_null.coords['x'])):
        for unit in range(len(v4_resp_ti_null.coords['unit'])):
            not_null = ~v4_resp_ti_null[unit,x,:].isnull()
            v4_resp_ti_null[unit, x, not_null] = np.random.permutation(
                                                 v4_resp_ti[unit, x, not_null].values)
    
    v4_resp_apc = v4_resp_apc.transpose('shapes','unit')
    for unit in range(len(v4_resp_apc_null.coords['unit'])):
        v4_resp_apc_null[:, unit] = np.random.permutation(v4_resp_apc[:, unit].values)

    null_v4 = process_V4(v4_resp_apc_null, v4_resp_ti_null, dmod)
    
    cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',]
    
    da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
    da = da.sel(unit=slice(0, None, 1)).squeeze()
    middle = np.round(len(da.coords['x'])/2.).astype(int)
    da_0 = da.sel(x=da.coords['x'][middle])
    indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
    layer_label = [da.coords['layer_label'].values[index] for index in sorted(indexes)]
       
    fns = [
    'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_null_analysis.p'
    ]

    alt = pd.concat([open_cnn_analysis(fns[0], layer_label)[-1], alt_v4], axis=0)
    init = open_cnn_analysis(fns[1], layer_label)[-1]
    shuf = open_cnn_analysis(fns[2], layer_label)[-1]
    null = pd.concat([open_cnn_analysis(fns[3], layer_label)[-1], null_v4], axis=0)
    cnn_an = pd.concat([alt, null, init, shuf ], 
              axis=0, keys=['resp', 's. resp', 'init. net', 's. layer wts'], names=['cond','layer_label','unit'])
    
fs = 9  
plt.figure(figsize=(8,6))
import matplotlib.gridspec as gridspec
conds = ['resp', 's. resp', 'init. net']
apc = cnn_an['apc'][cnn_an['k']<40]
m = 3
n = 3
gs = gridspec.GridSpec(m, n, width_ratios=[1,1,1],
                        height_ratios=[1,]*m, wspace=0.2)
ax_list = [plt.subplot(gs[pos]) for pos in range(m*n)]
labels = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.']
for ax, label in zip(ax_list, labels):
    ax.text(0, 1.1, label, transform=ax.transAxes,
      fontsize=fs, fontweight='bold', va='top', ha='right')
    

hist_pos = [0,3,6]
hist_dat_leg = []
hist_dat = [[apc.loc[cond].loc['v4'] for cond in ['resp', 's. resp']],]
hist_dat_leg.append({'title':'V4','labels':['Resp.', 'S. Resp.'], 
                     'fontsize':'xx-small','frameon':True,'loc':4 })

hist_dat.append([apc.loc[cond].drop('v4', level='layer_label') for cond in conds])
hist_dat_leg.append({'title':'CN all', 'labels':conds, 'fontsize':'xx-small',
                     'frameon':True,'loc':4})

layers_to_examine = ['conv1','conv2','conv5', 'fc6', 'fc7']
hist_dat.append([apc.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine])
hist_dat_leg.append({'title':'CN resp', 'labels':layers_to_examine, 
                    'fontsize':'xx-small' , 'frameon':True, 'loc':4,'markerscale':100})

for leg in hist_dat_leg:
    leg['fontsize'] = fs
    leg['labelspacing'] = 0
colors = ['r','g','b','m','c', 'k', '0.5']
for i, ax_ind in enumerate(hist_pos):
    ax = ax_list[ax_ind]
    for apc_vals, color in zip(hist_dat[i], colors):
        x = apc_vals.dropna().values
        y_c, bins_c = d_hist(ax, x, cumulative=True, color=color)   
    bins_c = np.concatenate([apc_vals.dropna().values for apc_vals in hist_dat[i]]).ravel()
    beautify(ax, spines_to_remove=['top','right'])
    #data_spines(ax, bins_c, y_c, mark_zero=[True, False], sigfig=2, fontsize=fs, 
    #            nat_range=[[0,1],[0,1]], minor_ticks=False, 
    #            data_spine=['bottom', 'left'], supp_xticks=[0.25,1,], 
    #            supp_yticks = [0.5,])
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xticklabels([0,0.25,0.5,0.75,1])

    ax.set_xlim(0,1)
    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_yticklabels([0,0.25,0.5,0.75,1])
    ax.set_ylim(0,1.1)
    leg = ax.legend(**hist_dat_leg[i])
    plt.setp(leg.get_title(),fontsize=fs)
    #ax.set_ylim(bottom=ax.get_ylim()[0] +ax.get_ylim()[0]*0.05, 
    #        top=ax.get_ylim()[1]+ax.get_ylim()[1]*0.05)
    ax.grid(axis='y')
    
    
example_cell_inds = [1,4,7]
v4 = cnn_an.loc['resp'].loc['v4']
v4_apc = v4[-v4['apc'].isnull()]
v4_apc[v4_apc['k']<40]
b_unit = v4_apc[v4_apc['cur_mean']>0.5]['apc'].argmax()
model = int(v4_apc['models'].loc[b_unit])
hi_curv_resp = v4_resp_apc.sel(unit=b_unit)
scatter_dat = [[hi_curv_resp, dmod.sel(models=model), 
                hi_curv_resp.coords['w_lab'].values],]

cn = cnn_an.loc['resp'].drop('v4', level='layer_label')
cn = cn[cn['k']<40]
cn_apc = cn[-cn['apc'].isnull()]     
b_unit = cn_apc[cn_apc['cur_mean']>0.5]['apc'].idxmax()
model = int(cn_apc['models'].loc[b_unit[0]].loc[b_unit[1]])

hi_curv_resp = da_0.sel(unit= b_unit[1])
model_resp = dmod.sel(models=model)
hi_curv_resp = hi_curv_resp.reindex_like(model_resp)
scatter_dat.append([hi_curv_resp, model_resp, b_unit])

cn = cnn_an.loc['resp'].drop('v4', level='layer_label')
cn = cn[cn['k']<40]
cn_apc = cn[-cn['apc'].isnull()]     
b_unit = cn_apc[cn_apc['cur_mean']<0.5]['apc'].idxmax()
model = int(cn_apc['models'].loc[b_unit[0]].loc[b_unit[1]])

hi_curv_resp = da_0.sel(unit=b_unit[1])
model_resp = dmod.sel(models=model)
hi_curv_resp = hi_curv_resp.reindex_like(model_resp)
scatter_dat.append([hi_curv_resp, model_resp, b_unit])
                

kw = {'s':1., 'linewidths':0, 'c':'r'}
for ax_ind, dat in zip(example_cell_inds, scatter_dat):
    ax = ax_list[ax_ind]
    x,y= scatter_lsq(ax, dat[0].values, dat[1].values, lsq=True,
                     mean_subtract=True, **kw)
    frac_var = np.corrcoef(x.T, y.T)[0,1]**2
    cartesian_axes(ax, x_line=True, y_line=True, unity=True)
    beautify(ax, spines_to_remove=['top','right', 'left','bottom'])
    ax.set_xticks([]);ax.set_yticks([]);
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y)+min(y)*0.05, max(y)+max(y)*0.05)
    if example_cell_inds[0]==ax_ind:
        ax.text(0, 0.5, 'Model',
                            transform=ax.transAxes, fontsize=fs,
                            va='center', ha='right', rotation='vertical')
    ax.set_title(' $R^2$:' +str(np.round(frac_var,2)), fontsize=fs)
    ax.text(.5, 0, 'Unit: ' +str(dat[2]),transform=ax.transAxes, 
            fontsize=fs, va='top', ha='center')
    
    ax = ax_list[ax_ind+1]
    im = plot_resp_on_shapes(ax, no_blank_image, dat[0], image_square=5)
    ax.set_xlabel('Curv $(\mu,\sigma)$ : ('+ str(np.round(dat[1].coords['cur_mean'].values,2))
                + ', ' + str(np.round(dat[1].coords['cur_sd'].values,2))+')'
                + '\n Ori $(\mu,\sigma)$ : ('+ str(np.round(np.rad2deg(dat[1].coords['or_mean'].values),0))
                +', '+ str(np.round(np.rad2deg(dat[1].coords['or_sd'].values),0))+')', fontsize=fs)
    cbar = plt.gcf().colorbar(im, ax=ax, ticks=[0, np.max(dat[0])],)
    cbar.ax.set_yticklabels(np.round([np.min(dat[0]), np.max(dat[0])],1)) # horizontal colorbarplt.gcf().colorbar(im)
ax_list[2].set_title('Rank Ordered\nUnit Shape Response', fontsize=8)

gs.tight_layout(plt.gcf())
plt.savefig(top_dir + '/analysis/figures/images/apc_figs.eps')


'''
pd.concat([apc.loc['resp'].drop('v4', level='layer_label')
                     for label in layer_label 
                     if 'conv' in label.astype(str)])

f, ax = plt.subplots(1,1,figsize=(4,4))
n, bins = np.histogram(v4_resp_apc.values.ravel())
d_cust_hist(ax, n, bins)
beautify(ax, spines_to_remove=['top','right'])
data_spines(ax, bins, n, mark_zero=[True, False], sigfig=1, fontsize=12, 
                nat_range=None, minor_ticks=False, data_spine=['bottom', 'left'])

var = v4_resp_apc.values
kw = {'s':6, 'linewidths':0, 'c':'k'}
x = np.random.normal(size=(20,1))
y = np.random.normal(size=(20,1)) + x
x, y = scatter_lsq(ax, x, y, lsq=True, mean_subtract=True, **kw)
beautify(ax, spines_to_remove=['top','right'])
data_spines(ax, x, y, mark_zero=[True, True], sigfig=1, fontsize=12, 
                nat_range=None, minor_ticks=False, data_spine=['bottom', 'left'])
cartesian_axes(ax, x_line=True, y_line=True, unity=True)


ax = plt.subplot(111)
x = v4_resp_apc.values.ravel()[1:100]
kde_dist(ax, x)
y, bins = d_hist(ax, x)
ax2 = ax.twinx()
y_c, bins_c = d_hist(ax2, x, cumulative=True)

x = v4_resp_apc.values.ravel()[100:200]
kde_dist(ax, x, color='r')
y, bins = d_hist(ax, x, bins=bins, color='r')
#ax2 = ax.twinx()
y_c, bins_c = d_hist(ax2, x, cumulative=True, color='r')


beautify(ax, spines_to_remove=['top',])
beautify(ax2, spines_to_remove=['top',])
data_spines_twinx([ax, ax2], x, [y, y_c], mark_zero=[True, False], sigfig=2, fontsize=12, 
                nat_range=None, minor_ticks=False, data_spine=['bottom', 'left', 'right'])
ax2.set_ylim(auto=True)
ax.set_ylim(bottom=ax.get_ylim()[0] +ax.get_ylim()[0]*0.05, 
            top=ax.get_ylim()[1]+ax.get_ylim()[1]*0.05)
ax2.set_ylim(bottom=ax2.get_ylim()[0] +ax2.get_ylim()[0]*0.05, 
            top=ax2.get_ylim()[1]+ax2.get_ylim()[1]*0.05)
'''




#data_spines(ax2, x, y, mark_zero=[True, False], sigfig=2, fontsize=12, 
#                nat_range=None, minor_ticks=False, data_spine=['bottom', 'left', 'right'])

#x = v4_resp_apc.values.ravel()[:1000]
#x = np.sort(x)
#y = np.array(range(len(x)))/float(len(x))
#
#from scipy import interpolate
#f = interpolate.PchipInterpolator(x, y).derivative()
#xnew = np.linspace(x.min(), x.max(), len(x))
#ynew = f(xnew)
#plt.plot(xnew, ynew)
##beautify(ax)
##beautify(ax2, spines_to_remove=['top', 'left'])
##kde_dist(ax, alt['apc'][alt['k']<42].dropna().values)

