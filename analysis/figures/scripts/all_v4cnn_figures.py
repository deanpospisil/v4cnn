#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:14:18 2016

@author: dean
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
os.chdir('/home/dean/Desktop/v4cnn/')
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn')
#sys.path.insert(0, top_dir + 'xarray/');
top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr
import pandas as pd
import apc_model_fit as ac
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp
import d_net_analysis as dn
import caffe_net_response as cf
from matplotlib.backends.backend_pdf import PdfPages

#%%

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
r        for spine in spines_to_remove:
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
        ax.plot([xlim[0], xlim[1]], [0, 0], color=more_grey,lw=1, alpha=0.5)
    if y_line:
        #ax.spines['left'].set_position('center')
        ax.plot([0, 0], [ylim[0], ylim[1]], color=more_grey,lw=1,alpha=0.5)
    
    if unity:
        ax.plot(xlim, xlim, color=more_grey, lw=1,alpha=0.5)

    
        
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
def d_cust_hist(ax, n, bins, color='k', lw=1):
    for a_n, a_bin in zip(n, bins):
        ax.plot([a_bin,a_bin],[0, a_n], color=color, lw=lw, linestyle=':', alpha=0.7)
    for i, a_n in enumerate(n):
        ax.plot([bins[i], bins[i+1]], [a_n, a_n],color=color, lw=lw)     
    
def d_hist(ax, x, bins='auto', alpha=0.5, color='k', lw=1, normed=True, cumulative=False):
     
    if bins=='auto':
        bins = np.round(np.sqrt(len(x))/2)
    if cumulative:
        if not all(np.isnan(x)):
            y_cum = np.array(range(1,len(x)+1))/float(len(x))
            ax.step(np.sort(x), y_cum, 
                    alpha=alpha,  color=color, lw=lw)
            ax.scatter([np.max(x)], [1,], color=color, marker='|', s=10)
            n = y_cum
            bins = np.sort(x)
        else:
            bins = None
            n = None
    else:
        n, bins = np.histogram(x, bins=bins, normed=True)
        d_cust_hist(ax, n, bins=bins, color=color, lw=lw)
        #n, bins, _ = ax.hist(x, bins=bins, color=color, histtype='step', 
        #                 alpha=alpha, lw=1, normed=normed)
        
    return n, bins
                 
def kde_dist(ax, x, bw=None, color='k'):
    x_grid = np.linspace(np.min(x), np.max(x), 1000)
    if bw == None:
        bw = np.std(x)*float(len(x))**(-1/5.)*2
    kde_skl = KernelDensity(bandwidth=bw)
    kde_skl.fit(x[:, np.newaxis])
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
    return data
    
def plot_resp_on_shapes(ax, imgStack, resp, image_square=19):
    resp_sc = resp.values
    resp_sc -= np.mean(resp_sc)
    max_dist = np.max(np.abs(resp_sc))
    imgStack = imgStack*resp_sc.reshape(362,1,1)
    #sort images
    sortStack = imgStack[list(reversed(np.argsort(resp_sc))),:,:]
    sortStack = np.array([imp.centeredCrop(img, 64, 64) for img in sortStack])
    im = vis_square(ax, sortStack[0:image_square**2])
    return im

def open_cnn_analysis(fn, layer_label):
    try:
        an=pk.load(open(fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(fn,'rb'))
    fvx = an[0].sel(concat_dim='r')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    return fvx, rf, cnn

def process_V4(v4_resp_apc, v4_resp_ti, dmod):
    ti = dn.ti_av_cov(v4_resp_ti, rf=None)
    apc = dn.ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                  dmod.chunk({}), fit_over_dims=None, 
                                    prov_commit=False)
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
               columns=['apc', ] + coords_to_take + [ 'k_stim',])
    v4 = pd.concat([v4pdti, v4pdapc])
    return v4

def r2_unbiased(y, x, n, m):
    y -= y.mean()
    x -= x.mean()
    b = np.dot(y, x)/np.dot(x, x)
    y_hat = b*x#our best fit of x to y in the 1-d model space
    res = y-b*x#the left over residual in (m-2)-d null space
    num = np.sum(y_hat**2)
    den = np.sum(y_hat**2) + np.sum(res**2)
    
    #R2 = (num)/(den)    
    R2_corrected = (num - (1./n))/((den - (m-2)*(1./n) - (1./n)))
    #R2_corrected = (num - (1./n))/((den - (m-1)*(1./n)))
    return R2_corrected
#%%
figure_num = [6, 7, 8, 9, 10, 4, 5, 1]
figure_num = [6, 10, 11, 7, 8, 9, 12, 4, 5, 1]
#shape image set up
img_n_pix = 100
max_pix_width = [80,]
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
center = np.ceil(img_n_pix/2.)
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict, 
                                                        base_stack, 
                                                        npixels=img_n_pix))
no_blank_image = trans_img_stack[1:]
a = np.hstack((range(14), range(18, 318)));a = np.hstack((a, range(322, 370)))
no_blank_image = no_blank_image[a]/255.
#%%

#%%
if sys.platform == 'linux2': 
    data_dir = '/loc6tb/'
else:
    data_dir = top_dir

goforit = True
#loading up all needed data
if 'cnn_an' not in locals() or goforit:
    
    v4_name = 'V4_362PC2001'
    v4_resp_apc = xr.open_dataset(data_dir + 'data/responses/v4cnn/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    file = open(data_dir + 'data/responses/v4cnn/v4_apc_109_neural_labels.txt', 'r')
    wyeth_labels = [label.split(' ')[-1] for label in 
                file.read().split('\n') if len(label)>0]
    v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
    fn = data_dir + 'data/models/' + 'apc_models_362.nc'
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp']
    dmod['models'] = range(dmod.shape[1])
    
    apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                      dmod.chunk({}), 
                                      fit_over_dims=None, 
                                      prov_commit=False)

    v4_resp_apc = v4_resp_apc - v4_resp_apc.mean('shapes')
    v4_resp_ti = xr.open_dataset(data_dir + 'data/responses/v4cnn/v4_ti_resp.nc')['resp'].load()
    alt_v4 = process_V4(v4_resp_apc, v4_resp_ti, dmod.load())

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
    
    cnn_names =['bvlc_reference_caffenetpix_width[ 8.4096606]_x_(64, 164, 51)_y_(114.0, 114.0, 1)PC370',]
    if sys.platform == 'linux2':
        da = xr.open_dataset(data_dir + 'data/responses/v4cnn/' + cnn_names[0] + '.nc')['resp']
    else:
        da = xr.open_dataset(data_dir + 'data/responses/v4cnn/' + cnn_names[0] + '.nc')['resp']
    da = da.sel(unit=slice(0, None, 1)).squeeze()
    middle = np.round(len(da.coords['x'])/2.).astype(int)
    da_0 = da.sel(x=da.coords['x'][middle])
    
    indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
    layer_label = [da.coords['layer_label'].values[index] for index in sorted(indexes)]
       
    fns = [
    'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_null_analysis.p'
    ]
    fns = [
    'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
    'blvc_caffenet_iter_1pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
    'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_null_analysis.p'
    ]
    
    
    fns = [ 'bvlc_reference_caffenetpix_width[ 8.4096606]_x_(64, 164, 51)_y_(114.0, 114.0, 1)PC370_analysis.p',
             'blvc_caffenet_iter_1pix_width[ 8.4096606]_x_(64, 164, 51)_y_(114.0, 114.0, 1)PC370_analysis.p',
             'bvlc_reference_caffenetpix_width[ 8.4096606]_x_(64, 164, 51)_y_(114.0, 114.0, 1)PC370_null_analysis.p',
            
            ]
#    fns = [
#    'bvlc_caffenet_reference_increase_wt_cov_random0.9pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
#    'blvc_caffenet_iter_1pix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
#    'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_null_analysis.p'
#    ]
    results_dir = data_dir + '/data/an_results/'
    alt = pd.concat([open_cnn_analysis(results_dir +  fns[0], layer_label)[-1], alt_v4], axis=0)
    init = open_cnn_analysis(results_dir + fns[1], layer_label)[-1]
    null = pd.concat([open_cnn_analysis(results_dir +fns[2], layer_label)[-1], null_v4], axis=0)
    cnn_an = pd.concat([alt, null, init], 
              axis=0, keys=['resp', 's. resp', 'init. net',], names=['cond','layer_label','unit'])
    
    
    corrected_v4_apc= []
    fn = '/loc6tb/data/models/apc_models_362.nc'
    apc = xr.open_dataset(fn)['resp']
    apc['models'] = range(apc.shape[1])
                                    
                                    

<<<<<<< HEAD
    #da_0 = da_0[apc.coords['shapes'].values]
    apc = apc - apc.mean('shapes')
    da_0n = apc/(apc.dot(apc, 'shapes')**0.5)
    
    
    v4 = xr.open_dataset('/loc6tb/data/responses/v4cnn/apc370t.nc')['resp']
    v4.coords['unit'] = range(109)
    v4s = v4[:,apc.coords['shapes'].values]
    def transform_exp(x, a, b):
        y = ((np.sqrt(a)*(1-0.5*b))**-1)*x**((1-0.5*b))
        return y
    for j, u in enumerate(v4s):
        y = u.values
        y = np.sum(y[...,200:800],-1)
        var = np.nanvar(y,-1)
        mu = np.nanmean(y,-1)
        n = np.sum(~np.isnan(y),-1).mean()
        vmr = np.nanmean(var/mu)

        y_expt = transform_exp(mu, a=vmr, b=1).T


        y_exptn = y_expt - np.mean(y_expt)
        y_exptn = y_exptn/(np.sum(y_exptn**2)**0.5)

        rind = np.nanargmax(np.dot(da_0n.values.T, y_exptn))
        corrected_v4_apc.append(r2_unbiased(y_expt, da_0n[..., rind].values, n=n, m=len(y_expt)))
        
        
=======
fn = results_dir +  fns[0]
try:
    an=pk.load(open(fn,'rb'), 
               encoding='latin1')
except:
    an=pk.load(open(fn,'rb'))
>>>>>>> 4b9ff218dfb66876a9eca6a5220ceffaa6d987fb
#%%
#labels_file = '/home/dean/caffe/' + 'data/ilsvrc12/synset_words.txt'
#labels = np.loadtxt(labels_file, str, delimiter='\t')
#labels[146]

#%%
layer_colors = cm.copper(np.linspace(0.1, 1, 8))
def vis_square(ax, data, padsize=0, padval=0):
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    ax.set_xticks([]);ax.set_yticks([])
    return data
def plot_resp_on_sort_shapes(ax, shapes, resp, top=25, fs=20, shrink=.5, colorbar=False):
    c_imgs = np.zeros(np.shape(shapes) + (4,))
    respsc = (resp - resp.min())
    respsc = respsc/respsc.max()
    
    scale = cm.cool(respsc)
    resp_sort_inds = np.argsort(resp)[::-1]
    
    for i, a_color in enumerate(scale):
        c_imgs[i, np.nonzero(shapes[i])[0], np.nonzero(shapes[i])[1],:] = a_color
    
    im = ax.imshow(np.tile(respsc,(2,1)), cmap=cm.cool, interpolation='nearest')
    if colorbar:
        cbar = ax.get_figure().colorbar(im, ax=ax, shrink=shrink, ticks=[0,1]) 
        cbar.ax.set_ylabel('',rotation='horizontal', fontsize=fs/1.5, ha='center')
        cbar.ax.tick_params(axis='both', which='both',length=0)
#    cbar = ax.get_figure().colorbar(im, ax=ax, shrink=shrink, 
#            ticks=[np.min(respsc), np.max(respsc)], aspect=10)
#    cbar.ax.set_yticklabels([]) 
    #cbar.ax.set_ylabel('Normalized\nResponse', rotation='horizontal', fontsize=fs/1.5, ha='left')
    
    data = vis_square(ax, c_imgs[resp_sort_inds][range(0,8) + range(-9,-1)])
    ax.imshow(data, interpolation='nearest')
    #beautify(ax, ['top','right','left','bottom'])
    return data
fs = 9  
figsize= (5,5)
plt.figure(figsize=figsize)
import matplotlib.gridspec as gridspec
conds = ['resp', 's. resp', 'init. net']
cnn_an['apc'] = cnn_an['apc']

apc = cnn_an['apc'][cnn_an['k']<42].dropna()
m = 3
n = 3
gs = gridspec.GridSpec(m, n, width_ratios=[1,.7,.5],
                        height_ratios=[1,]*m, wspace=0)
ax_list = [plt.subplot(gs[pos]) for pos in range(m*n)]

hist_pos = [0,3,6]
hist_dat_leg = []

v4apc1 = cnn_an.loc['resp'].loc['v4']['apc'].dropna().copy(deep=True)
v4apc2 = cnn_an.loc['resp'].loc['v4']['apc'].dropna().copy(deep=True)
v4apc2[...] = corrected_v4_apc
hist_dat = [[v4apc1,v4apc2**0.5]]

#hist_dat = [[pd.DataFrame(corrected_v4_apc)**0.5],]
hist_dat_leg.append({'labels':['Resp.', 'S. Resp.'], 
                     'fontsize':'xx-small', 'frameon':True, 'loc':4 })

hist_dat.append([apc.loc[cond].drop('v4', level='layer_label') for cond in conds])
hist_dat_leg.append({'labels':['Resp.', 'S. Resp.', 'Untrained'], 'fontsize':'xx-small',
                 'frameon':True,'loc':4})

layers_to_examine = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
#layers_to_examine = ['relu1','pool1', 'norm1', 'relu2','pool2', 'norm2', 'pool5', 
#                     'relu3','relu4','relu5', 'relu6','relu7',]
hist_dat.append([apc.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine] + 
    [v4apc1,v4apc2**0.5])
#hist_dat.append([apc.loc['resp'].drop('v4', level='layer_label').loc[layer]
#                 for layer in layers_to_examine])
hist_dat_leg.append({'title':'CN resp', 'labels':layers_to_examine, 
                    'fontsize':'xx-small' , 'frameon':True, 'loc':4,'markerscale':100})

    
for leg in hist_dat_leg:
    leg['fontsize'] = fs
    leg['labelspacing'] = 0
for i, ax_ind in enumerate(hist_pos):
    ax = ax_list[ax_ind]
    if i==0:
        colors = ['r','g','b','m','c', 'k', '0.5']
    elif i==1:
        colors = ['k','g','b','m','c', 'k', '0.5']
    else:
        colors = list(layer_colors)
        colors.append([1, 0, 0, 0], )
        colors.append([1, 0, 1, 0], )
    for apc_vals, color in zip(hist_dat[i], colors):
        x = apc_vals.dropna().values
        if i==2:
            lw=1
        else:
            lw=2
        y_c, bins_c = d_hist(ax, x, cumulative=True, color=color, 
                             alpha=0.75, lw=lw)   
    bins_c = np.concatenate([apc_vals.dropna().values for apc_vals in hist_dat[i]]).ravel()
    beautify(ax, spines_to_remove=['top', 'right'])
    ax.set_xticks([0,.25,0.5,.75,1])
    ax.set_xticklabels([0, 0.25,0.5,.75, 1], fontsize=10)
    ax.spines['left'].set_bounds(0,1)
    ax.set_xlim(0.1,0.85)
    
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0,  ' ', 1], fontsize=10)
    ax.set_ylim(0, 1.1)
    if not (ax_ind==hist_pos[-1]):
        print(ax_ind, hist_pos[-1])
        #leg = ax.legend(**hist_dat_leg[i])
        #plt.setp(leg.get_title(),fontsize=fs)
    else:
        v4 = plt.Line2D((0,1), (0,0), color='r',lw=3)
        early = plt.Line2D((0,1), (0,0), color=cm.copper(0), lw=3)
        late = plt.Line2D((0,1), (0,0), color=cm.copper(1), lw=3)
        #ax.legend([v4, early, late], ['V4','CN Layer 1', 'CN Layer 8'],
        #fontsize=fs , frameon=True, loc=4, labelspacing=0)
    ax.grid()

ax_list[0].set_title('Cumulative Distribution', fontsize=12) 
ax_list[0].set_ylabel('Fraction < r', labelpad=0, fontsize=12) 
ax_list[0].text(0.6, 0.1, 'V4', transform=ax_list[0].transAxes, fontsize=12)
#ax_list[0].text(0.05,0.6, 'Shuffled', transform=ax_list[0].transAxes, color='g', rotation=80)
#ax_list[0].text(0.4,0.6, 'Unshuffled', transform=ax_list[0].transAxes, color='r',rotation=45)


#ax_list[3].text(0.5, 0.1, 'CNN all layers', transform=ax_list[3].transAxes)
#ax_list[3].text(0.24,0.75, 'Untrained', transform=ax_list[3].transAxes, color='b', rotation=60)

#ax_list[6].text(0.5, 0.1, 'CNN by layer', transform=ax_list[6].transAxes)
colors = list(cm.copper(np.linspace(0, 1, 8)))
colors.append('r')
layers_to_examine.append('V4')
layer_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8', 'V4']
spaces = np.linspace(0.85, 0.02, len(layer_names))

#for name, color, space in zip(layer_names, colors, spaces):
#    ax_list[6].text(0.05, space, name, transform=ax_list[6].transAxes,
#           color=color, fontsize=7, bbox=dict(facecolor='white', ec='none',
#                                              pad=0))
ax_list[6].set_xlabel('APC fit r', labelpad=0, fontsize=12)
example_cell_inds = [1,4,7]
v4 = cnn_an.loc['resp'].loc['v4']
v4_apc = v4[-v4['apc'].isnull()]
b_unit = v4_apc[v4_apc['cur_mean']>0.5]['apc'].argmax()
model = int(v4_apc['models'].iloc[b_unit])
hi_curv_resp = v4_resp_apc.sel(unit=b_unit)
scatter_dat = [[hi_curv_resp, dmod.sel(models=int(model)), 
                hi_curv_resp.coords['w_lab'].values],]

cn = cnn_an.loc['resp'].drop('v4', level='layer_label')
cn = cn[cn['k']<42]
cn_apc = cn[-cn['apc'].isnull()] 
b_unit = cn_apc[cn_apc['cur_mean']>0.5].loc['conv2']['apc'].idxmax()

b_layer_unit = 113
b_layer_name = 'conv2'
b_unit = da_0[:,(da_0.layer_unit == b_layer_unit) 
    * (da_0.layer_label == b_layer_name)].unit.values
model = cn_apc['models'].loc[b_layer_name].loc[b_unit]


hi_curv_resp = da_0.sel(unit=b_unit).squeeze()
model_resp = dmod.sel(models=model.values[0].astype(int)).squeeze()
hi_curv_resp = hi_curv_resp.reindex_like(model_resp)
scatter_dat.append([hi_curv_resp, model_resp, ('conv2', b_unit)])

cn = cnn_an.loc['resp'].drop('v4', level='layer_label')
cn = cn[cn['k']<42]
cn_apc = cn[-cn['apc'].isnull()]     
b_unit = cn_apc[cn_apc['cur_mean']<0.5]['apc'].idxmax()

b_layer_unit = 3591
b_layer_name = 'fc7'
b_unit = da_0[:,(da_0.layer_unit == b_layer_unit) 
 * (da_0.layer_label == b_layer_name)].unit.values

model = int(cn_apc['models'].loc[b_layer_name].loc[b_unit])

hi_curv_resp = da_0.sel(unit = b_unit).squeeze()
model_resp = dmod.sel(models = model).squeeze()
hi_curv_resp = hi_curv_resp.reindex_like(model_resp)
scatter_dat.append([hi_curv_resp, model_resp, b_unit])
                

kw = {'s':2., 'linewidths':0, 'c':'k'}
colorbar = 1
for ax_ind, dat in zip(example_cell_inds, scatter_dat):
    ax = ax_list[ax_ind]
    x,y = scatter_lsq(ax, dat[0].values, dat[1].values, lsq=True,
                     mean_subtract=True, **kw)
    frac_var = np.corrcoef(x.T, y.T)[0,1]
    print(frac_var)
    print(dat[1].coords)
    cartesian_axes(ax, x_line=True, y_line=True, unity=True)
    beautify(ax, spines_to_remove=['top','right', 'left','bottom'])
    ax.set_xticks([]);ax.set_yticks([]);
    ax.set_xlim(min(x), max(x))
    
    ax.set_ylim(min(y)+min(y)*0.05, max(y)+max(y)*0.05)
    if ax_ind>1:
        print(dat[0].layer_label.values)
        ax.set_title(str(dat[0].layer_label.values) + ' ' + str(int(dat[0].layer_unit.values)))
    else:
        ax.set_title(dat[0].w_lab.values)
        print(dat[0].w_lab.values)
        
    if example_cell_inds[0]==ax_ind:
        #ax.text(0, 0.5, 'Model',
        #                    transform=ax.transAxes, fontsize=fs,
        #                    va='center', ha='right', rotation='vertical')
        ax.set_ylabel('Model',labelpad=0)
        ax.set_xlabel('Unit', labelpad=0)
#    params = 'Curv. $(\mu=$' +  str(np.round(dat[1].coords['cur_mean'].values,2))\
#    +', $\sigma=$'+ str(np.round(dat[1].coords['cur_sd'].values,2)) + ')'\
#    +'\n \nOri. $(\mu=$'+ str(np.round(np.rad2deg(dat[1].coords['or_mean'].values)))\
#    +', $\sigma=$' + str(np.round(np.rad2deg(dat[1].coords['or_sd'].values),0)) + ')' 
    #if ax_ind==1:
#        ax.set_title('Example units')
#        ax.text(0.5, 0.3, '$r=$' +str(np.round(frac_var, 3)), 
#                transform=ax.transAxes, fontsize=10)
#        ax.text(0.35, 0.07, params, 
#                transform=ax.transAxes, fontsize=7, linespacing=0.5)
    #else:
#        ax.text(0.6, 0.2, '$r=$' +str(np.round(frac_var, 3)), 
#                transform=ax.transAxes, fontsize=10)
#        ax.text(0.35, -0.1, params, linespacing=.5,
#                transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='white',
#                                                              ec='none', alpha=0.8))        
#    ax.text(.5, 0, 'Unit: ' +str(dat[2]),transform=ax.transAxes, 
#            fontsize=fs, va='top', ha='center')
    
    ax = ax_list[ax_ind+1]
    
    data = plot_resp_on_sort_shapes(ax, no_blank_image, dat[0], top=16, fs=fs, 
                                    shrink=0.75, colorbar=colorbar)
    if colorbar:
        colorbar=0
    ax.imshow(data)
#    if example_cell_inds[0]==ax_ind:
#        ax.set_title(' APC Model Params'
#                +'\nCurv. $(\mu=$' +  str(np.round(dat[1].coords['cur_mean'].values,2))
#                +', $\sigma=$'+ str(np.round(dat[1].coords['cur_sd'].values,2)) + ')'
#                +'\nOri. $(\mu=$'+ str(np.round(np.rad2deg(dat[1].coords['or_mean'].values)))
#                +', $\sigma=$' + str(np.round(np.rad2deg(dat[1].coords['or_sd'].values),0)) + ')'
#                , fontsize=fs)
#    else:
#        ax.set_title('Curv. $(\mu=$' +  str(np.round(dat[1].coords['cur_mean'].values,2))
#                +', $\sigma=$'+ str(np.round(dat[1].coords['cur_sd'].values,2)) + ')'
#                +'\nOri. $(\mu=$'+ str(np.round(np.rad2deg(dat[1].coords['or_mean'].values)))
#                +', $\sigma=$' + str(np.round(np.rad2deg(dat[1].coords['or_sd'].values),0)) + ')'
#                , fontsize=fs)
    if ax_ind==1:
        #ax.set_xlabel(str(dat[2]) , labelpad=3) 
        ''
    elif ax_ind==4:
        #ax.set_xlabel(str(dat[2][0])+ ' ' +str(dat[2][1]), labelpad=3) 
        ''
    else:
        #ax.set_xlabel(str(dat[2][0])+ ' ' +str(dat[2][1]), labelpad=3)
        ''
    #ax.set_ylim(500,2)
ax_list[2].text(-0.1, 1.2, 'Preferred Shapes', transform=ax_list[2].transAxes, fontsize=12)
#ax_list[2].set_position([0.9, 0.9, 1, 1])
import matplotlib 
c = matplotlib.transforms.Bbox([[0.71, 0.75], [0.91, .9]])
#ax_list[2].set_position(c)

#left=None, bottom=None, right=None, top=None,
#ax_list[2].set_title('Preferred Shapes', fontsize=12, labelpad=10)
gs.tight_layout(plt.gcf())
labels = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.']
for ax, label in zip(ax_list, labels):
    ax.text(-0, 1.1, label, transform=ax.transAxes,
      fontsize=fs+2, fontweight='bold', va='top', ha='right')

plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'+
            str(figure_num[0]) + '_apc_figs_v4cnn.pdf', bbox_inches='tight',
            dpi=500)

#%%
plt.figure()
import d_net_analysis as na

y_nm = 'bvlc_reference_caffenetpix_width[ 8.4096606]_x_(114.0, 114.0, 1)_y_(64, 164, 51)PC370.nc'
x_nm = cnn_names[0] + '.nc'

ti = []
k = []
for net_name in [y_nm, x_nm]:
    da = xr.open_dataset(data_dir + '/data/responses/v4cnn/'+ net_name)['resp'].squeeze()
    k.append(na.kurtosis_da(da))
    ti.append(na.ti_in_rf(da, stim_width=32))
non_k_var = (k[0][1]<42) * (k[1][1]<42) * (k[0][0]<6) * (k[1][0]<6)
ti_x_f = ti[1][non_k_var]
ti_y_f = ti[0][non_k_var]
ti_x_f = np.ma.masked_invalid(ti_x_f)
ti_y_f = np.ma.masked_invalid(ti_y_f)

n_intervals = 10.
interval_space = 1./n_intervals
intervals = np.linspace(0, 1-interval_space, n_intervals)

c_means_x = []
c_means_y = []
c_sd_x = []
c_sd_y = []

for interval in intervals:
    cond_y = ti_y_f[(ti_x_f>interval)*(ti_x_f<=interval+interval_space)]
    cond_x = ti_x_f[(ti_y_f>interval)*(ti_y_f<=interval+interval_space)]
    c_means_x.append(cond_x.mean())
    c_means_y.append(cond_y.mean())
    c_sd_x.append(np.percentile(cond_x[np.isfinite(cond_x)], [5, 95]))
    c_sd_y.append(np.percentile(cond_y[np.isfinite(cond_y)], [5, 95]))


c_sd_y_err = np.ma.abs((np.array(c_sd_y) - np.array(c_means_y).reshape(int(n_intervals),1)).T)
c_sd_x_err = np.ma.abs((np.array(c_sd_x) - np.array(c_means_x).reshape(int(n_intervals),1)).T)

#%%
import scipy.stats as st
y = np.array(ti_y_f)
x = np.array(ti_x_f)
tf = ~(np.isnan(x) + np.isnan(y))
y = y[tf]
x = x[tf]
print(st.linregress(x, y))


plt.axis('square')
fs = 12
plt.xlim(-0.1, 1)
plt.ylim(-0.1, 1)
plt.errorbar(intervals+interval_space/2, c_means_y, yerr=c_sd_y_err, color='k',fmt='.', lw=1)
plt.scatter(ti_x_f, ti_y_f, s=2,  c='c', edgecolors='none')
plt.xlabel('TI X', fontsize=fs)
plt.ylabel('TI Y', fontsize=fs)
plt.tight_layout()
plt.xticks(np.linspace(0,1,11))
plt.gca().set_xticklabels(['0','','0.2','','0.4','','0.6', '','0.8','', '1'])
plt.yticks([0,0.2,0.4,0.6, 0.8, 1])
plt.gca().set_yticklabels(['0','0.2','0.4','0.6', '0.8', '1'])
plt.plot([0,1],[0,1])
plt.grid(True, which='both')
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'+str(figure_num[1])+ 
            '_ti_x_vs_y_all.pdf', bbox_inches='tight')


#%%
ti_x = ti[1]
ti_y = ti[0]
layer_names = ['Conv2','Relu2', 'Pool2', 'Norm2', 'Conv3', 'Relu3','Conv4', 
               'Relu4','Conv5', 'Relu5', 'Pool5', 'FC6', 'Relu6', 'FC7', 'Relu7', 'FC8',]
layers = da.coords['layer'].values
layer_labels = da.coords['layer_label'].values
n_plots = len(np.sort(np.unique(layers))[4:-1])
plt.figure(figsize=(8,8))
for i, layer in enumerate(np.sort(np.unique(layers))[4:-1]):
    plt.subplot(4,4, i+1)
    layer_ind = layer == layers
    layer_label = layer_labels[layer_ind][0]
    plt.scatter(np.array(ti_x[layer_ind]), np.array(ti_y[layer_ind]),s=4, edgecolors='none')
    
    plt.plot([0,1],[0,1], color='k')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    plt.axis('square')
    if i==0:
        plt.xlabel('TI X')
        plt.ylabel('TI Y')
        plt.title(layer_names[i])
        plt.xticks([0,0.5,1])
        plt.yticks([0,0.5,1])
        plt.xlim(0,1);plt.ylim(0,1)

    else:
        plt.title(layer_names[i])
        plt.xticks([]);plt.yticks([])
        plt.xticks([0,0.5,1])
        plt.yticks([0,0.5,1])
        plt.xlim(0,1);plt.ylim(0,1)
        plt.gca().set_yticklabels(['','',''])
        plt.gca().set_xticklabels(['','',''])

plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/' + str(figure_num[2]) 
            + '_ti_x_vs_y.pdf' )

#%%
m = 1
n = 2
plt.figure(figsize=(8/(1.3*1.1), 4/(1.5*1.1)))
gs = gridspec.GridSpec(m, n, width_ratios=[1,]*n,
                        height_ratios=[1,]*m) 

ax_list = [plt.subplot(gs[pos]) for pos in range(m*n)]
labels = ['A.', 'B.', 'C.', 'D.',]
for ax, label in zip(ax_list, labels):
    ax.text(-0.1, 1.2, label, transform=ax.transAxes,
      fontsize=fs, fontweight='bold', va='top', ha='right')
          
ti_v4 = alt_v4[~alt_v4['ti_av_cov'].isnull()]['ti_av_cov']
ex_cell_inds = [ti_v4.argmax()[1]]
ex_cell_ti = [ti_v4.loc['v4'].loc[ind] for ind in ex_cell_inds]
ex_cell_resp = [v4_resp_ti.sel(unit=ind).dropna('shapes',how='all').dropna('x', how='all')
                     for ind in ex_cell_inds]
v4_ex = [cell for cell in zip(ex_cell_ti, ex_cell_resp, ex_cell_inds)]
props = {'facecolor':'w', 'boxstyle':'round', 'alpha':1, 'clip_on':True}

v4_ex_plt_inds = [0,1]
for plt_ind, ex_cell in zip(v4_ex_plt_inds, v4_ex):
    ax = ax_list[plt_ind]
    ex_cell_resp = ex_cell[1]*(1/.3)
    rf = (ex_cell_resp).mean('shapes')
    corr = np.corrcoef(ex_cell_resp)
    middle = rf.argmax()
    corr_slice=corr[:, middle]
    #ax.locator_params(axis='x', nbins=len(rf.values), tight=True);
    ax.set_xticklabels(['-1','-1','0','1','2'])
    
    
    ax.plot(rf.values, color='g')
    ax.scatter(range(0,4), rf.values, color='g')
    ax2 = ax.twinx()
    ax2.locator_params(axis='y', nbins=5, tight=True);
    ax2.locator_params(axis='x', nbins=len(rf.values), tight=True);

    ax2.plot(corr_slice, color='r')
    ax2.scatter(range(0,4), corr_slice, color='r')

    beautify([ax,ax2], spines_to_remove=['top',]);

    ax.set_ylim(0, rf.max() + rf.max()*0.1)
    ax.text(0.5, 0.5, 'TI=' + str(np.round(ex_cell[0],2)), color='k', 
            rotation='horizontal', transform=ax.transAxes, ha='center', 
            va='center', fontsize=12, 
            bbox=props)
    #ax.set_title('TI = '+str(np.round(ex_cell[0],2)))
    
    ax2.set_ylim(0,1.1);
    ax2.set_yticks([0,0.5,1])
    #ax.set_xlim(-0.1,len(rf.values)-1+0.1)
    
    ax.set_yticks([0,12,24])
    ax.set_ylim([0,28])

    
    if plt_ind == 0:
        #ax2.legend(['$$', ],loc=10, frameon=False)
        ax.set_xlabel('Position', labelpad=1)
        #ax.legend(['RF',], loc=8, frameon=False)
        ax.set_ylabel('RF\n$\mu$ spk/s', rotation='horizontal', labelpad=30, 
                      color='g', ha='center',va='center', fontsize=14)
        ax2.set_ylabel('$R$', rotation='horizontal', ha='center',va='center', 
                       labelpad=12, color='r', fontsize=14)
    ax2.spines['right'].set_bounds(0,1)
    ax.spines['right'].set_bounds(0,ax.get_ylim()[1]*.9)
    
    ax = ax_list[plt_ind-1]
    a = ex_cell_resp.sel(x=2).values
    b = ex_cell_resp.sel(x=3).values
    scatter_lsq(ax, a, b, lsq=0, mean_subtract=0, **{'s':1})
    ax.locator_params(nbins=10, tight=True);
    ax.axis('equal');
    ax.set_xticks(np.round([0, max(a)]), minor=False)
    ax.set_yticks(np.round([0,max(b)]), minor=False)
    #for cell 33 needed to correct this to x=2 x=1 which corresponds to .16 fraction rf, and 0 fraction rf
    beautify(ax);
    fs=12
    ax.set_xlabel('Pos. ' +str(0)+'\nspk/s',labelpad=10, fontsize=fs,va='top',);
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_ylabel('Pos. ' +str(-1)+'\nspk/s', rotation=0,labelpad=10, color='k', 
                  ha='right',va='center', fontsize=fs);
    ax.yaxis.set_label_coords(-0.35, 0.35)
    ax.xaxis.set_label_coords(.5, -0.35)
    ax.set_aspect(1)
    ax.plot([-2,166],[-2,166], color='0.5')
    #ax.set_xlim([0, 166])
    #ax.set_ylim([0, 166])
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/'
            +str(figure_num[3])+'_ti_v4.pdf', bbox_inches='tight')


#%%
# just example cells
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
    return resp_av_cov_da, in_rf

ti, inrf = ti_in_rf(da, 32)
#%%

m = 3
n = 2
plt.figure(figsize=(6,6))
gs = gridspec.GridSpec(m, n, width_ratios=[1,]*n, height_ratios=[1,]*m) 

ax_list = [plt.subplot(gs[pos]) for pos in range(m*n)]
labels = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.']
for ax, label in zip(ax_list, labels):
    ax.text(0, 1.3, label, transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')
    
rf = open_cnn_analysis(data_dir + 'data/an_results/' + fns[0], layer_label)[1]
cor = open_cnn_analysis(data_dir + 'data/an_results/' + fns[0], layer_label)[0]
av_cors = cor.groupby('layer').mean('unit')
av_rfs = rf.groupby('layer').mean('unit')

ex_avg_layer = [17, 12, 4][::-1]
#ex_avg_layer = [ b'conv1', b'fc7'][::-1]

ex_inds = [1, 3, 5]  

for layer, ex_ind in zip(ex_avg_layer, ex_inds):
    ax = ax_list[ex_ind]

    av_cor = av_cors[layer]
    av_rf = av_rfs[layer]
    av_rf = av_rf/av_rf.max()
    
    ax.plot(av_rf.coords['x'].values, av_rf.values, alpha=1, 
             lw=2, color='g')
    print(av_rf.values)
    ax.plot(av_cor.coords['x'].values, av_cor.values, alpha=1, lw=2, 
            color='r')

    ax.set_ylim(-0.2, 1);
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels([])


    ax.set_xlim(64, 164)
    ax.set_xticks([64, 114, 164])
    ax.set_xticklabels([])
    if ex_ind ==1:
        ax.set_title('Layer Average\n', fontsize=17)

    beautify([ax,], spines_to_remove=['top', 'right'],); 
    

ti_cnn = cnn_an[~cnn_an['ti_in_rf'].isnull()]['ti_in_rf'].loc['resp']
ex_cell_inds = [('conv2', 387), ('conv5', 2545), ('fc7', 12604),]

ex_cell_inds_layer_unit = [('conv2', 113), ('conv4', 369), ('fc7', 3591),]
ex_cell_inds_unit = []
for ex_cel_ind in ex_cell_inds_layer_unit:
    b_unit = da_0[:,(da_0.layer_unit == ex_cel_ind[1]) 
     * (da_0.layer_label == ex_cel_ind[0])].unit.values
    ex_cell_inds_unit.append((ex_cel_ind[0], int(b_unit))) 
    

ex_cell_tis = [ti_cnn.loc[ind[0]].loc[ind[1]] for ind in ex_cell_inds_unit]
ex_cell_cors = [cor.sel(unit=ind[1]) for ind in ex_cell_inds_unit]
ex_cell_rfs = [rf.sel(unit=ind[1]) for ind in ex_cell_inds_unit]
cn_ex = [cell for cell in zip(ex_cell_tis, ex_cell_cors, ex_cell_rfs, ex_cell_inds_unit)]
ex_cell_inds = [0,2,4]
ti_leg_pos = [[0.97,0.8],[0.97,0.8],[0.97,0.25]]
for ex_cell,  ex_cell_ind, layer, ti_leg in zip(cn_ex,  ex_cell_inds, 
                                                ['conv2', 'conv4', 'fc7'], 
                                                ti_leg_pos):
    ax = ax_list[ex_cell_ind]
    cell_label = ex_cell[-1]
    cell_ti = ex_cell[0]
    
    ex_cell_rf = ex_cell[2]
    ex_cell_rf /= ex_cell_rf.max()
    ex_cell_cor = ex_cell[1]
    ex_cell_ti = ex_cell[1]    
    ax.plot(ex_cell_rf.coords['x'].values, ex_cell_rf, color='g', lw=2)
    ax.plot(ex_cell_rf.coords['x'].values, ex_cell_cor, color='r',lw=2, alpha=1)
    ax.plot(ex_cell_rf.coords['x'].values, inrf[int(ex_cell[1].coords['unit'].values)])
    
    ax.set_xlim()
    beautify([ax, ax2], spines_to_remove=['top', 'right']);
    ax.set_ylim(-0.2,1)
    ax.set_xticks([64, 114, 164])
    ax.set_xticklabels([-50,0,50])
    ax.set_xlim([64, 164])

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', ' ', '1'], fontsize=12)

    #ax.text(str(cell_label) + '. TI='+ str(np.round(cell_ti,2)))
    ax.text(-0.38, 0.5, layer, color='k', rotation='vertical',
                transform=ax.transAxes, ha='center', va='center', fontsize=16)
    props = {'facecolor':'w', 'boxstyle':'round', 'alpha':1, 'clip_on':True}
    ax.text(ti_leg[0],ti_leg[1], 'TI=' + str(np.round(cell_ti,2)), color='k', 
            rotation='horizontal', transform=ax.transAxes, ha='right', 
            va='center', fontsize=12, 
            bbox=props)

    if ex_cell_ind==0:
        #ax2.set_ylabel('$R^2$', rotation=0, labelpad=0, ha='left', va='center')
        #ax.set_ylabel('scaled\n$\mu$\nresp.', rotation=0, labelpad=15,ha='center',va='center')
        ax.set_xlabel('Position (pixels)', labelpad=1, fontsize=12)
        ax.set_title('Example Units\n', fontsize='16')
        ax.text(-0.20,0.5, 'Correlation', color='r',rotation='vertical', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.text(-0.28,0.5, 'Receptive Field', color='g', rotation='vertical',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
    else:
        ax.set_xticklabels([])

#    if ex_cell_ind ==0:
#        ax2.legend(['$R^2$', ],loc='upper left', frameon=False, handletextpad=0, markerfirst=False)
#        ax.legend(['RF',], loc='center left', frameon=False,handletextpad=0, markerfirst=False)

    #ax.set_yticklabels([])
ax_list[0].text(0.5,1.01, 'Unit ' + str(ex_cell_inds_layer_unit[0][1]), ha='center', va='bottom',
                transform=ax_list[0].transAxes, fontsize=12, fontstyle='italic')
#ax_list[0].annotate('', xy=(101, 0.05), xytext=(101, 0.4), ha='center',
#            arrowprops=dict(facecolor='black', shrink=0.05),zorder=1, fontsize=8)
ax_list[4].text(.5,1.01, 'Unit ' + str(ex_cell_inds_layer_unit[2][1]), ha='center', va='bottom',
                transform=ax_list[4].transAxes, fontsize=12, fontstyle='italic')
ax_list[2].set_title('Unit ' + str(ex_cell_inds_layer_unit[1][1]), fontstyle='italic')
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'+str(figure_num[4])+
            '_ti_example_and_avg_v4cnn.pdf', bbox_inches='tight')
#%%
cnn_name = cnn_names[0]
lims = 300       
da = xr.open_dataset(data_dir + 'data/responses/v4cnn/' + cnn_name + '.nc')['resp'].squeeze()
pos1 = 114
pos2 = 120
pos3 = 102
plt.figure(figsize=(4,2))
plt.subplot(121)
plt.scatter(da[..., 497].sel(x=pos1), da[..., 497].sel(x=pos2), s=3, edgecolors='none')
plt.xlim(-lims,lims);plt.ylim(-lims,lims);
plt.xticks([-lims,0,lims]);plt.yticks([-lims,0,lims])
r = np.corrcoef(da[..., 497].sel(x=pos1).reindex_like(model_resp), 
            da[..., 497].sel(x=pos2).reindex_like(model_resp))[0,1]
plt.title('r= ' + str(np.round(r,2)))
plt.xlabel('Response at\nPosition 0')
plt.ylabel('Response at\nPosition +6')
plt.plot([-300,300],[-300,300], color='k', alpha=0.5)
plt.axis('equal')


plt.subplot(122)
plt.scatter(da[..., 497].sel(x=pos1), da[..., 497].sel(x=pos3), s=3, edgecolors='none')
plt.xlim(-lims,lims);plt.ylim(-lims,lims);
plt.xticks([-lims,0,lims]);plt.yticks([-lims,0,lims])
plt.gca().set_xticklabels([]);plt.gca().set_yticklabels([])
r = np.corrcoef(da[..., 497].sel(x=pos1).reindex_like(model_resp), 
            da[..., 497].sel(x=pos3).reindex_like(model_resp))[0,1]
plt.plot([-300,300],[-300,300], color='k', alpha=0.5)
plt.title('r= ' + str(np.round(r, 2)))
plt.ylabel('Response at\nPosition -6')
plt.axis('equal')
plt.xlabel('Response at\nPosition 0')
plt.tight_layout()

plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'+str(figure_num[4])+
            'a_ti_example_and_avg_v4cnn.pdf', bbox_inches='tight')


#%%
from scipy import io

da = io.loadmat(data_dir + 'data/responses/v4cnn/cadieu_109.mat')['all']
resp = np.reshape(da, (65, 109, 368), order='F')
da = xr.DataArray(resp, dims=['x', 'unit','shapes'], coords=[range(0, 65*4, 4), range(109),  range(368)])
da = da.transpose('unit', 'x', 'shapes')
#cadieu_ti = na.ti_in_rf(da, stim_width=64).to_pandas()


plt.figure(figsize=(4,4))
gs = gridspec.GridSpec(2,1, width_ratios=[1,]*1,
                        height_ratios=[1,]*2) 
ax_list = [plt.subplot(gs[pos]) for pos in range(2)]
labels = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.']
for ax, label in zip(ax_list, labels):
    ax.text(-0.1, 1.15, label, transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')
ti_cnn = cnn_an[~(cnn_an['ti_in_rf'].isnull())]
#ti_cnn = cnn_an
ti_cnn = ti_cnn[(ti_cnn['k_pos']>2)&(ti_cnn['k_pos']<40)]['ti_in_rf']

hist_pos = [1,0]
hist_dat_leg = []
hist_dat = []
layers_to_examine = ['conv2','conv3','conv4', 'conv5', 'fc6', 'fc7', 'fc8']

hist_dat.append([ti_cnn.loc['init. net'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine])
hist_dat_leg.append({'labels':['CN', 'CN init.'], 
                     'fontsize':'xx-small','frameon':False,'loc':(-0.2,1) })

    
cat = pd.read_csv(top_dir + 'data/responses/PositionData_Yasmine/TXT_category', delimiter=' ')
cat = cat['c'].values
cat_true = cat==1
v4_ti_av_cov = cnn_an.loc['resp'].loc['v4']['ti_av_cov']
v4_ti_vex  = v4_ti_av_cov[:80][cat_true]

#layers_to_examine = ['relu1','relu2','relu3','relu4', 'relu5', 'relu6', 'relu7', 'fc8']
hist_dat.append([ti_cnn.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine] + 
               [v4_ti_vex,] )
#                 [cnn_an.loc['resp'].loc['v4']['ti_av_cov'],] + [cadieu_ti,])
hist_dat_leg.append({'title':'CN layers', 'labels':layers_to_examine, 
                    'fontsize':'xx-small' , 'frameon':False, 'loc':(-0.3,1)})
fs= 8



lw=2
for leg in hist_dat_leg:
    leg['fontsize'] = fs
    leg['labelspacing'] = 0
    leg['loc'] = (-0.35,1)
colors = ['g', 'r', '0.5', 'm', 'b', 'c', 'k']
for i, ax_ind in enumerate(hist_pos):
    ax = ax_list[ax_ind]
    if i ==0:
        colors = layer_colors[1:]
        lw = 1.5
    else:
        colors = list(layer_colors[1:])
        colors.append(np.array([1,0,0,1]))
        colors.append(np.array([0,0,1,1]))

    for ti_val, color in zip(hist_dat[i], colors):
        x = ti_val.dropna().values
        y_c, bins_c = d_hist(ax, x, cumulative=True, color=color, alpha=1, lw=lw)

    bins_c = np.concatenate([ti_val.dropna().values for ti_val in hist_dat[i]]).ravel()
    beautify(ax, spines_to_remove=['top','right'])
    #data_spines(ax, bins_c, y_c, mark_zero=[True, False], sigfig=2, fontsize=fs, 
               # nat_range=[[0,1],[0,1]], minor_ticks=False, 
                #data_spine=['bottom', 'left'], supp_xticks=[0.5,], 
                #supp_yticks = [0.5,])
    for spine_name in ['left', 'bottom']:
        ax.spines[spine_name].set_bounds(0,1)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0,0.5,1])
    ax.set_xticklabels(['0', '0.5', '1'], fontsize=10)

    ax.set_ylim(-0.1,1.001)
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0', ' ', '1'], fontsize=10)
    #ax.set_ylabel('Fraction Units', fontsize=14)
    #leg = ax.legend(**hist_dat_leg[i],ncol=3)
    #plt.setp(leg.get_title(),fontsize=fs)
    ax.set_ylim(bottom=ax.get_ylim()[0] + ax.get_ylim()[0]*0.05, 
            top=ax.get_ylim()[1]+ax.get_ylim()[1]*0.05)
    ax.grid(axis='y')

    #ax.set_xlabel('Translation Invariance', fontsize=14)
ax_list[0].set_xticklabels([])
ax_list[0].set_ylabel('Fraction Units', labelpad=0, fontsize=10)
ax_list[1].set_xlabel('Translation Invariance', labelpad=0, fontsize=14)
ax_list[1].set_title('Untrained AlexNet')

layer_names = [ 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8', 'V4', 'Cadieu']

spaces = np.linspace(0.9, 0.15, len(layer_names))
colors = list(colors)
for name, color, space in zip(layer_names, colors, spaces):
    print(name)
    print(color)
    ax_list[0].text(0.05, space, name, transform=ax.transAxes,
           color=color, fontsize=12, bbox=dict(facecolor='white', ec='none', pad=0))
    

plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'
            +str(figure_num[5])+'_ti_training_and_layer.pdf')

#%%
colors = np.array([[226,128,9,1],[190,39,45,1], [127,34, 83,1], [ 119, 93, 153,1], 
          [54, 58, 100,1], [157,188,88,1], [75,135,71,1],[ 59, 88,62,1], [0,0,0,1]])
colors = colors / np.array([[255,255,255,1]])
colors = list(layer_colors[1:])
colors.append(np.array([1,0,0,1]))


cat = pd.read_csv(top_dir + 'data/responses/PositionData_Yasmine/TXT_category', delimiter=' ')
cat = cat['c'].values
cat_true = cat==1
v4_ti_av_cov = cnn_an.loc['resp'].loc['v4']['ti_av_cov']
v4_ti_vex = v4_ti_av_cov[:80][cat_true]


plt.figure(figsize=(5,3))
gs = gridspec.GridSpec(1,2, width_ratios=[1,2],
                        height_ratios=[1,]*1) 
ax_list = [plt.subplot(gs[pos]) for pos in range(2)]
labels = ['A.', 'B.',]
for ax, label in zip(ax_list, labels):
    ax.text(0, 1.3, label, transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')
ax = ax_list[0]
n_samples=100
apc_cor = apc

v4 = cnn_an.loc['resp'].loc['v4']
v4_apc = v4[-v4['apc'].isnull()]['apc']
v4_apc= v4apc2**0.5

for layer, color in zip(['fc7', 'conv2',],[colors[6],colors[1]]):
    ax.scatter(ti_cnn.loc['resp'].loc[layer][:n_samples], 
               apc_cor.loc['resp'].loc[layer][:n_samples], 
               color=color, s=1, alpha=1)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect(1)
beautify(ax)
#ax.grid()
ax.set_xlim(0,1.01)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xticklabels([0,0.5,1])
ax.set_yticklabels([0, ' ' ,1])
ax.set_ylabel('APC', labelpad=30, rotation=0, va='center',ha='left')
ax.set_xlabel('TI', labelpad=5)
plt.tight_layout()

v4ness = ((1-ti_cnn.loc['resp'])**2 + (1-apc_cor.loc['resp'])**2)**0.5
(layer,num)= v4ness.argmin()
y = apc_cor.loc['resp'].loc[layer].loc[num]
x = ti_cnn.loc['resp'].loc[layer].loc[num]
ax.scatter(x, y, color=colors[6], marker='*', s=4)


#best_v4_ti = cnn_an.loc['resp'].loc['v4']['ti_av_cov'].max()
best_v4_ti = v4_ti_vex.max()

best_v4_apc = v4_apc.max()
ax.scatter(best_v4_ti, best_v4_apc, color='r', marker='x',s=4)
ax.legend(['Conv2', 'FC7', 'Best AN', 'Best V4'], fontsize=5, loc=3, 
          labelspacing = 0, scatterpoints=1)

#avg_ti_v4 = cnn_an.loc['resp'].loc['v4']['ti_av_cov'].mean()
avg_ti_v4 = v4_ti_vex.mean()
avg_apc_v4 = apc_cor.loc['resp'].mean()
ax.plot([0, avg_ti_v4], [avg_apc_v4, avg_apc_v4], color='grey', lw=0.5)
ax.plot([avg_ti_v4, avg_ti_v4], [0, avg_apc_v4], color='grey', lw=0.5)
#ti_cnn.loc['resp'].drop('v4', level='layer_label').max()

ax = ax_list[1]
layers_to_examine = ['conv2','conv3','conv4', 'conv5', 'fc6', 'fc7', 'fc8']
hist_dat = []
hist_dat = [v4ness.drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine]
for v4ness_val, color in zip(hist_dat, colors):
    x = v4ness_val.dropna().values
    y_c, bins_c = d_hist(ax, x, cumulative=True, color=color, alpha=1, lw=1.4) 
    ax.scatter([np.min(x),], [0,], color=color, marker='|')

ax.set_xlim(1,0.15)
ax.set_xticks([1,0.5,.15])
ax.set_xticklabels(['.15', '0.5', '1'][::-1])

ax.set_ylim(1,-0.1)
ax.set_aspect(0.5)
ax.set_yticks([1,0.5,0])
ax.set_yticklabels(['0', ' ', '1'])
beautify(ax)
ax.grid()
ax.set_xlabel('Distance to APC=1 & TI=1', labelpad=2)

nbins = 100
v4_apc_hist, bins = np.histogram(v4_apc.values, 
                           density=True, bins=nbins, range=(0,1))
#v4_ti_hist, bins = np.histogram(cnn_an.loc['resp'].loc['v4']['ti_av_cov'].dropna().values, 
#                          density=True, bins=nbins, range=(0,1))
v4_ti_hist, bins = np.histogram(v4_ti_vex, 
                          density=True, bins=nbins, range=(0,1))

v4_apc_hist /= len(v4_apc_hist)
v4_ti_hist /= len(cnn_an.loc['resp'].loc['v4']['ti_av_cov'].dropna().values)

joint_hist = np.expand_dims(v4_apc_hist,1)[::-1] * np.expand_dims(v4_ti_hist, 0)
x, y = np.meshgrid(bins[1:][::-1], bins[1:])
dist = (x**2 + y**2)**0.5
dist_sort = np.argsort(dist.ravel())
cum_hist = np.cumsum(joint_hist.ravel()[dist_sort])
cum_hist = cum_hist/max(cum_hist)
end = np.sum((1-best_v4_apc)**2 + (1-best_v4_ti)**2)**0.5
dist_sort_val = dist.ravel()[dist_sort]
plt.plot(dist_sort_val[dist_sort_val>end], cum_hist[dist_sort_val>end], color='r')
ax.scatter([end,], [0,], color='r', marker='|')

layer_names = [ 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8', 'V4']
spaces = np.linspace(0.9, 0.15, len(layer_names))
for name, color, space in zip(layer_names, colors, spaces):
    ax.text(0.05, space, name, transform=ax.transAxes,
           color=color, fontsize=7, bbox=dict(facecolor='white', ec='none', pad=0))
    

plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'+
             str(figure_num[6]) + '_v4_ness.pdf')
##%%
#un_inds = xr.open_dataset(data_dir + '/data/models/apc_models_362_16X16.nc')['resp'].coords['shapes'].values
#layers_to_examine = ['conv1', 'relu1', 'norm1',  'conv2', 'fc6', 'prob']
#layer_names = ['Conv1', 'Relu1', 'Norm1',  'Conv2', 'FC6', 'Prob']
##layers_to_examine = 'all'
#name = 'bvlc_reference_caffenetAPC362_pix_width[64.0]_pos_(64.0, 164.0, 51).nc'
#cnn = [xr.open_dataset(data_dir + 'data/responses/v4cnn/' + name)['resp'].sel(x=114), ]
#
#name = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(114.0, 114.0, 1)_amp_(100, 255, 2).nc'
#cnns = [xr.open_dataset(data_dir + 'data/responses/v4cnn/' + name)['resp'].sel(amp=amp) for amp in [255, 100]] + cnn
#art = [xr.open_dataset(data_dir + 'data/responses/v4cnn/' + name)['resp'],]   
#art = art[0].isel(amp=1)
#
#name = 'bvlc_reference_caffenet_nat_image_resp_371.nc'
#cnn = [xr.open_dataset(data_dir + 'data/responses/v4cnn/' + name)['resp'],]   
#nat = cnn[0]
#cnns = cnns + cnn
#cnns = [cnns[0], cnns[2], cnns[1], cnns[-1]]
#
#
#rat = (nat.max('shapes')/art.max('shapes')).squeeze()
#rat[rat==np.inf] = 0
#rat[rat==-np.inf] = 0
#
#a = rat.groupby('layer').mean()[3:]
#a.plot()
#a.coords['layer_label']
#%%
n_plot = len(layers_to_examine)
plt.figure(figsize=(3/1.5,n_plot*1.5/1.5))

gs = gridspec.GridSpec(n_plot,1) 
ax_list = [plt.subplot(gs[pos]) for pos in range(n_plot)]

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#from scipy.stats.kde import gaussian_kde
the_range = []
for i, a_lay in enumerate(layers_to_examine):
    all_nets = []
    for a_cnn in cnns:
        all_lays = a_cnn.coords['unit'].layer_label.values.astype(str)
        var = a_cnn[...,all_lays==a_lay].values.ravel()
        all_nets.append(var)
    the_range.append(np.abs([np.max(np.array(all_nets)), np.min(np.array(all_nets))]).max())
colors = ['b', 'c', 'g', 'r']  
for i, a_lay in enumerate(layers_to_examine):
    ax = ax_list[i]

    for j, a_cnn in enumerate(cnns[::-1]):

        all_lays = a_cnn.coords['unit'].layer_label.values.astype(str)
        var = a_cnn[...,all_lays==a_lay].values[1:,:][un_inds].ravel()
        n, bins = np.histogram(var, bins=100, normed=False) 
        
        n =  n/float(len(var));
        ax.plot(bins[1:], np.convolve(gaussian(np.linspace(-1,1,20), 0, 0.15), 
                n, mode='same'), color=colors[j])
    ax.semilogy(nonposy='clip')
    ax.set_ylim(10./len(var), 1)
    ax.set_xlim(-the_range[i], the_range[i])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([0, ax.get_xlim()[1]])
    ax.set_xticklabels([' ', int(np.round(ax.get_xlim()[1],0))])
    ax.set_title(layer_names[i],)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax.set_yticks([1, 1./100])
    ax.set_yticklabels(['',''])
    ax.tick_params('y', length=0, width=0, which='minor')

cond_names = ['Standard Shapes', 'Dimmer Shapes', 'Larger Shapes', 'Photographs']
ax_list[0].legend(ax_list[0].lines[::-1],
        cond_names, loc=(-0.3, 1.5), frameon=0, 
                    fontsize=7, markerfirst=False)

ax_list[-1].set_ylabel('%')
#ax_list[0].set_xlabel('Response', labelpad=0)
ax_list[-1].text(0.2,-.8, 'Response', transform=ax_list[-1].transAxes, color='k', rotation=0)
ax_list[-1].set_yticklabels(['100', '1'])
ax_list[-1].set_xticklabels([0, 1])

plt.tight_layout(h_pad=0.2)
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'
            +str(figure_num[7])+'_dynamic_range.pdf',
            bbox_inches='tight')
#%%
#kurtosis v4 example
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
def kurtosis(da):
    da = da.transpose('shapes', 'unit')
    mu = da.mean('shapes')
    sig = da.reduce(np.nanvar, dim='shapes')
    k = (((da - mu)**4).sum('shapes', skipna=True) / da.shapes.shape[0])/(sig**2)
    return k
n_plot = 3
plt.figure(figsize=(n_plot*2.5, 1*3.6))

gs = gridspec.GridSpec(2,n_plot) 
ax_list = [plt.subplot(gs[pos]) for pos in range(3)]
labels = ['A.', 'B.','C.', 'D.', 'E.']
for ax, label in zip(ax_list, labels):
    ax.text(-0.1, 1.3, label, transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')

v4_name = 'V4_362PC2001'
<<<<<<< HEAD
v4_resp_apc_b = xr.open_dataset(data_dir + 'data/responses/v4cnn/' + v4_name + '.nc')['resp'].load()
v4_resp_apc_b = v4_resp_apc.transpose('shapes', 'unit')
=======
v4_resp_apc_b = xr.open_dataset(data_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
#v4_resp_apc_b = v4_resp_apc.transpose('shapes', 'unit')
>>>>>>> 4b9ff218dfb66876a9eca6a5220ceffaa6d987fb
k_apc = kurtosis(v4_resp_apc_b).values

ax = ax_list[0]
ax.set_xlabel('Normalized Firing Rate')
ax.set_ylabel('Fraction Responses', labelpad=0)
ax.set_xticks([0,1])
ax.set_title('Example Response\nDistributions V4')
n_bins = 10
var = v4_resp_apc_b[:, np.argmax(k_apc)]
print(var)
n_samps = len(var)
ax.hist(v4_resp_apc_b[:, np.argmax(k_apc)], bins=n_bins, histtype='step', 
        weights=[1./n_samps,]*n_samps,  range=[0,1], color='r', log=True)
ax.hist(v4_resp_apc_b[:, np.argsort(k_apc)[len(k_apc)//2]], histtype='step',
        bins=n_bins, range=[0,1], weights=[1./n_samps,]*n_samps, color='b')
ax.hist(v4_resp_apc_b[:, np.argmin(k_apc)], histtype='step', bins=n_bins, 
                      range=[0,1], weights=[1./n_samps,]*n_samps, color='g')


ax.legend(['Max.', 'Median', 'Min.'], loc=[0.05,0], markerfirst=True, 
            fontsize=6, frameon=False, columnspacing=0, title='Kurtosis')
ax.tick_params('y', which = 'both', right=0)

ax.set_yticks([1, 0.1, 0.01])
ax.set_yticklabels([1, 0.1, 0.01])
ax.xaxis.set_label_coords(0.5, -0.3)
ax.yaxis.set_label_coords(-0.3, 0.5)

####
ax = ax_list[1]
ax.hist(k_apc, bins=30, histtype='step', weights=[1./len(k_apc),]*len(k_apc), 
                                                  color='k')
ax.set_xlabel('Kurtosis');

ax.set_ylabel('Fraction Units', labelpad=0);
ax.set_xticks([0,  42])
ax.set_yticks([0,  .5])
ax.set_yticklabels(['0',  .5])

ax.xaxis.set_label_coords(0.5, -0.3)
ax.yaxis.set_label_coords(-0.2, 0.5)
ax.set_ylim(-0.0,0.5)
ax.set_title('Kurtosis Distribution\nV4')
ax.annotate(s='',xy=(42,0), xytext=(42,0.05),
            arrowprops=dict(ec ='red', facecolor='red', headwidth=6),
            zorder=1, )
ax.annotate(s='',xy=(3.9,0), xytext=(3.9,0.05),
            arrowprops=dict(ec ='blue', facecolor='blue', headwidth=6),
            zorder=1, )
ax.annotate(s='',xy=(2.3,0), xytext=(2.3,0.05),
            arrowprops=dict(ec ='green', facecolor='green', headwidth=6),
            zorder=1,)


#n =  n/float(len(var));
#ax.plot(bins[1:], np.convolve(gaussian(np.linspace(-1,1,20), 0, 0.1), n, mode='same'))
#plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig1_kurtosis_example_v4.pdf')
n_bins = 20


ax = ax_list[2]
k_apc = alt.drop('v4', level='layer_label')['k_stim'].dropna().values
ax.hist(k_apc, bins=n_bins, histtype='step', weights=[1./len(k_apc),]*len(k_apc), 
        log=True, color='k', range=[0,370])
ax.set_xlabel('Kurtosis');
ax.set_xticks([0,  42,370])
ax.set_yticks([0.01, .1, 1])
ax.set_yticklabels([ '0.1','0.01', '1'])


ax.set_ylim(-0.0,1)
ax.set_ylabel('Fraction Units')
ax.tick_params('y', which = 'both', right=0)
ax.xaxis.set_label_coords(0.5, -0.3)
ax.yaxis.set_label_coords(-0.3, 0.5)
#ax.set_ylim(10**(-4.5),1)

#ax = ax_list[3]
layers_to_examine = ['relu1','pool1', 'norm1', 'relu2','pool2', 'norm2', 'pool5', 
                     'relu3','relu4','relu5', 'relu6','relu7',]
var = np.concatenate([alt['k_stim'].iloc[layer==all_lays].dropna().ravel() 
                        for layer in layers_to_examine])
n_samps = len(var)
ax.hist(var, bins=n_bins, histtype='step', weights=[1./n_samps,]*n_samps,
         color='m', range=[0,370], log=True,)

layers_to_examine = ['conv1','conv2','conv3','conv4','conv5', 'fc6','fc7','fc8']
all_lays= alt.index.get_level_values(0)

var = np.concatenate([alt['k_stim'].iloc[layer==all_lays].dropna().ravel() 
                        for layer in layers_to_examine])
n_samps = len(var)
ax.hist(var, bins=n_bins, histtype='step', weights=[1./n_samps,]*n_samps,
         color='c',log=True, range=[0,370])
ax.set_xlim(0, 371)
ax.set_xticks([0,  42, 370])
ax.set_yticks([1, 0.1, 0.01])
ax.set_yticklabels([1, 0.1, 0.01])
#ax.set_ylim(10**(-4.5),1)
ax.tick_params('y', which = 'both', right=0)
ax.legend(['All Layers', 
           'Rectified', 
           'Unrectified'], loc=[0.3, 0.55], 
            fontsize=7.6, frameon=False, columnspacing=0, borderpad=0)
ax.set_title('Kurtosis Distribution\nCNN')

[[layer] for layer in alt.index.levels[0]]

[beautify(an_ax) for an_ax in ax_list]
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'
            +str(figure_num[8])+'_kurtosis.pdf')
#%%
import pickle
import itertools
flatten_iter = itertools.chain.from_iterable
def factors(n):
    return set(flatten_iter((i, n//i) 
                for i in range(1, int(n**0.5)+1) if n % i == 0))
goforit = True      
if 'a' not in locals() or goforit:
    with open(top_dir + 'nets/netwts.p', 'rb') as f:    
        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)
            
import matplotlib.gridspec as gridspec
m = 8
n = 12
wts = a[0][1]
plt.figure(figsize=(6,4),)
plot_id = np.arange(0, m*n).reshape(m, n)
gs = gridspec.GridSpec(8, 12)
gs.update(wspace=0.0, hspace=0.0)

p_num = -1
for filt in wts:
    filt = filt - filt.min()
    filt = filt/filt.max()
    p_num += 1 
    ax = plt.subplot(gs[p_num])
    ax.imshow(np.swapaxes(np.swapaxes(filt,0,2),0,1), interpolation = 'nearest')        
    ax.set_xticks([]);ax.set_yticks([]);
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'
            +str(1)+ '_1stfilters.pdf')

#%%
def boot_strap_se(a, bstraps=1000):
    stats = []
    for ind in range(bstraps):
        resample = np.random.randint(0, high=np.shape(a)[0], size=np.shape(a)[::-1])
        stats.append([np.mean(a[col, i]) for i, col in enumerate(resample)])
    return np.percentile(np.array(stats), [1,99], axis=0)

def cor2(a,b):
    if len(a.shape)<=1:
        a = np.expand_dims(a,1)
    if len(b.shape)<=1:
        b = np.expand_dims(b,1)
    a -= a.mean(0);
    b -= b.mean(0)
    a /= np.linalg.norm(a, axis=0);
    b /= np.linalg.norm(b, axis=0);
    corrcoef = np.dot(a.T, b)       
    return corrcoef
 
#v4 fit to CNN and APC
v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(data_dir + 'data/responses/v4cnn/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
file = open(data_dir + 'data/responses/v4cnn/v4_apc_109_neural_labels.txt', 'r')
wyeth_labels = [label.split(' ')[-1] for label in 
            file.read().split('\n') if len(label)>0]
v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
fn = data_dir + 'data/models/' + 'apc_models_362.nc'

if 'apc_fit_v4' not in locals():
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()
    apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                      dmod.chunk({}), 
                                      fit_over_dims=None, 
                                      prov_commit=False)**2

cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
            'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)']
colors = ['r','g','b','m','c', 'k', '0.5']
from sklearn.model_selection import ShuffleSplit
X = np.arange(362)
cv_scores = []
model_ind_lists = []
models = []
for cnn_name in cnn_names:
    da_temp = xr.open_dataset(data_dir + 'data/responses/v4cnn/' + cnn_name + '.nc')['resp']
    da_temp = da_temp.sel(unit=slice(0, None, 1)).squeeze()
    middle = np.round(len(da_temp.coords['x'])/2.).astype(int)
    da_0_temp = da_temp.sel(x=da_temp.coords['x'][middle])
    da_0_temp = da_0_temp.sel(shapes=v4_resp_apc.coords['shapes'].values)
    models.append(da_0_temp)
models.append(dmod)

#cross_val fit
n_splits = 50
for model in models:
    ss = ShuffleSplit(n_splits=n_splits, test_size=1/5.,
        random_state=0)
    cv_score = []
    model_ind_list = []
    for train_index, test_index in ss.split(X):
        cor_v4_model = cor2(model.values[train_index], v4_resp_apc.values[train_index])
        cor_v4_model[np.isnan(cor_v4_model)] = 0
        model_sel = cor_v4_model.argmax(0)
        cor_v4_model_cv = np.array([cor2(v4_resp_apc[test_index, i], model[test_index, model_ind])
                            for i, model_ind in enumerate(model_sel)]).squeeze()
        model_ind_list.append(model_sel)
        cor_v4_model_cv[np.isnan(cor_v4_model_cv)] = 0
        cv_score.append(cor_v4_model_cv)
    cv_scores.append(cv_score)
    model_ind_lists.append(model_ind_list)
model_ind_lists_cv = np.array(model_ind_lists)
cor_v4_models_cv = np.array(cv_scores)


#direct fit
model_ind_lists = []
cor_v4_models = []
for model in models:
    cor_v4_model = cor2(model.values, v4_resp_apc.values)
    cor_v4_model[np.isnan(cor_v4_model)] = 0
    model_sel = cor_v4_model.argmax(0)
    model_cor = cor_v4_model.max(0)
    
    cor_v4_models.append(model_cor)
    model_ind_lists.append(model_sel)

    
model_ind_lists_dirfit = np.array(model_ind_lists)
cor_v4_models_dirfit = np.array(cor_v4_models)


cv_scores = np.array(cv_scores)
mean_scores = cv_scores.mean(1)
#bsci_scores= np.array([boot_strap_se(cv_score) for cv_score in cv_scores])
bsci_scores= np.array([np.percentile(np.array(cv_score), [5,95], axis=0) for cv_score in cv_scores])

bsci_scores = bsci_scores - np.expand_dims(mean_scores,1)

ax_list=[]
plt.figure(figsize=(4,4))
ax = plt.subplot(221)
ax_list.append(ax)
ax.locator_params(nbins=5)
ax.set_title('V4 Models Comparison\n')
x = mean_scores[0]
y = mean_scores[2]
xsd = bsci_scores[0]
ysd = bsci_scores[2]
ax.errorbar(x, y, yerr=np.abs(ysd), xerr=np.abs(xsd), fmt='o', 
            alpha=0, markersize=0, color='r', ecolor='0.5')
colors= np.array(['k',]*len(x))
colors[((np.abs(x-y)-np.max(np.abs(ysd),0))>0) & 
       ((np.abs(x-y)-np.max(np.abs(xsd),0))>0)] = 'r'

red_above = (((np.abs(x-y)-np.max(np.abs(ysd),0))>0) & 
((np.abs(x-y)-np.max(np.abs(xsd),0))>0) & ((y-x)>0))       

red_above_ind = [i for i, val in enumerate(red_above) if val]
n_red_above = np.sum(red_above)

red_below = (((np.abs(x-y)-np.max(np.abs(ysd),0))>0) & 
((np.abs(x-y)-np.max(np.abs(xsd),0))>0) & ((y-x)<0))
red_below_ind = [i for i, val in enumerate(red_below) if val]

n_red_below = np.sum(red_below)

ax.scatter(x,y, color=colors, s=3)
#ax.scatter(x, y, alpha=0.5, s=2)
ax.plot([0,1],[0,1], color='0.5')
#ax.set_xlabel('Trained Net')
ax.set_ylabel('APC\nR',labelpad=0)
ax.yaxis.set_label_coords(-0.4, 0.5)

ax.set_ylim(0,1)
ax.set_xlim(0,1)
ax.set_xticks([0,0.5,1])
ax.set_xticklabels([])
ax.set_yticks([0, 0.5, 1])
plt.grid()
ind_bfitcnn = cor_v4_models_dirfit[0].argsort()[-1]
ind_bfit_trut = (cor_v4_models_dirfit[0] - cor_v4_models_dirfit[1]).argsort()[-1]


ax.annotate(s='a9501', xy=[x[ind_bfitcnn], y[ind_bfitcnn]+0.03], xytext=[x[ind_bfitcnn], y[ind_bfitcnn]+0.3],
            arrowprops={'shrink':0.0, 'headwidth':10, 'frac':0.1, 'facecolor':'none'})
ax.annotate(s='b8302', xy=[x[ind_bfit_trut], y[ind_bfit_trut]+0.03], xytext=[x[ind_bfit_trut], y[ind_bfit_trut]+0.3],
           arrowprops={'shrink':0.0, 'headwidth':10, 'frac':0.1, 'facecolor':'none'})
beautify(ax)

ax = plt.subplot(223)
ax_list.append(ax)

ax.locator_params(nbins=5)
x = mean_scores[0]
y = mean_scores[1]
xsd = bsci_scores[0]
ysd = bsci_scores[1]

#ax.scatter(x, y, alpha=0.5, s=2)
ax.errorbar(x, y, yerr=np.abs(ysd), xerr=np.abs(xsd), fmt='o', 
            alpha=0, markersize=0, color='r', ecolor='0.5')
colors= np.array(['k',]*len(x))
colors[((np.abs(x-y)-np.max(np.abs(ysd),0))>0) & 
       ((np.abs(x-y)-np.max(np.abs(xsd),0))>0)] = 'r'
       
       
ax.scatter(x,y, color=colors, s=3)
ax.plot([0,1],[0,1], color='0.5')
ax.set_ylim(0,1)
ax.set_xlim(0,1)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
beautify(ax)
ax.annotate(s='a9501', xy=[x[ind_bfitcnn], y[ind_bfitcnn]+0.03], xytext=[x[ind_bfitcnn], y[ind_bfitcnn]+0.3],
            arrowprops={'shrink':0.0, 'headwidth':10, 'frac':0.1, 'facecolor':'none'},    
        ha='right')
ax.annotate(s='b8302', xy=[x[ind_bfit_trut]+0.03, y[ind_bfit_trut]], xytext=[x[ind_bfit_trut]+0.3, y[ind_bfit_trut]],
            arrowprops={'shrink':0.0, 'headwidth':10, 'frac':0.1, 'facecolor':'none'})
ax.set_xlabel('R\nTrained Net',labelpad=5)

ax.set_ylabel('Untrained Net', labelpad=12)
ax.yaxis.set_label_coords(-0.52, 0.5)
plt.tight_layout()
plt.grid()

labels = ['A.', 'B.']
for ax, label in zip(ax_list, labels):
    ax.text(-0.35, 1.12, label, transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/13_apc_vs_cnn.pdf')




#%%
import matplotlib as mpl
frac_of_image = 0.25

def cur_or_dict(s, norm=True):
    cs = s[:, 1]*1j + s[:, 0]
    downsamp = 1
    if norm:
        adjust_c = 3.8 # cuvature values weren't matching files I got so I scaled them
        a = {'curvature': 
        -((2. / (1 + np.exp(-0.125 * dc.curve_curvature(cs)* adjust_c)))-1)[::downsamp],
        'orientation': 
        ((np.angle(dc.curveAngularPos(cs)))% (np.pi * 2))[::downsamp]}
    else:

        a = {'curvature': 
        - dc.curve_curvature(cs)[::downsamp],
        'orientation': 
        ((np.angle(dc.curveAngularPos(cs)))% (np.pi * 2))[::downsamp]}
        
        
    return a
    
def match_ori_max_cur(shape_dict_list_pasu, ws):  
    or_dense = ws['orientation']
    or_pasu = shape_dict_list_pasu[shape_id]['orientation']
    cur_dense = ws['curvature']
    
    or_dif = abs(np.expand_dims(or_pasu, axis=1)-np.expand_dims(or_dense,axis=0))
    min_or_dif = np.pi/20
    close_bool = list(or_dif < min_or_dif)
    close_inds = [np.array(np.where(a_close_bool)).T for a_close_bool in close_bool]
    
#    #select based on closeness to original Pasu, or and cur.
#    match_loc = [close_ind_set[np.argmin(
#                    abs(cur_dense[close_ind_set] - cur_pasu_point))][0].astype(int)
#                    for close_ind_set, cur_pasu_point 
#                    in zip(close_inds, cur_pasu)]
                    
    match_loc = [close_ind_set[np.argmax(abs(cur_dense[close_ind_set]))][0]
                for close_ind_set
                in close_inds]
                
    return match_loc


    
with open(data_dir + '/data/models/PC370_params.p', 'rb') as f:
    shape_dict_list_pasu = pickle.load(f)
cmap = cm.bwr 
mat = l.loadmat(top_dir + '/img_gen/'+ 'PC3702001ShapeVerts.mat')
s = np.array(mat['shapes'][0])
s = [shape[:-1,:] for shape in s]
s = dc.center_boundary(s)

normed = True
shape_id = 105
rect_len = 8

shape_dict_list_dense = (cur_or_dict(ashape / np.max(np.abs(ashape)), norm=normed)
                         for ashape in s)
ws = itertools.islice(shape_dict_list_dense, shape_id, shape_id+1).next()
dense_val = np.array([ws['curvature'], 
                      ws['orientation']]).T

orig_val = np.array([shape_dict_list_pasu[shape_id]['curvature'], 
                     shape_dict_list_pasu[shape_id]['orientation']]).T


match_loc_orig = match_ori_max_cur(shape_dict_list_pasu, ws)
schematic_gaussian = True


ashape = s[shape_id]
norm = mpl.colors.Normalize(vmin=-1.,vmax=1.)

subsamp = 1
to_roll = len(dense_val[::subsamp, 1]) - np.argmax(dense_val[::subsamp, 1])

x = np.roll(dense_val[::subsamp, 1], to_roll)
y = np.roll(dense_val[::subsamp, 0], to_roll)
n_pts = len(x)
line_segs = [[[x[ind], y[ind]], [x[ind+1], y[ind+1]]] for ind in range(n_pts-1)]
curv = dense_val[:, 0]

left = -np.pi/4
right = 2*np.pi
if schematic_gaussian:
    from matplotlib.patches import Ellipse
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111)
    m_curv = 0
    sd_curv = 0.15
    m_ori = np.deg2rad(270)
    sd_ori = .7
    
    e = Ellipse(xy=[m_ori, m_curv], width=sd_ori*2, height=sd_curv*2)
    e.set_facecolor([0.5,0.5,0.5,0.5])
    ax.add_artist(e)

    curv_x = np.linspace(-1,1,100)  
    curv_gaus = gaussian(curv_x, m_curv, sd_curv)
    or_x = np.linspace(0, 2*np.pi, 100)
    or_gaus = gaussian(or_x, m_ori, sd_ori)
    fs=12
    plt.plot( or_x, (0.3*or_gaus-1.05), color='k')
    plt.plot([m_ori, m_ori], [-1, m_curv-sd_curv], color='k')
    plt.plot([m_ori-sd_ori, m_ori-sd_ori], [-1, m_curv],  color='0.5')
    plt.plot([m_ori+sd_ori, m_ori+sd_ori], [-1, m_curv], color='0.5',)
    ax.text(m_ori + 0.1, -0.5, '$\mu_a$', fontsize=fs)
    ax.text(m_ori+sd_ori+0.1, -0.5, '$\sigma_a$', fontsize=fs)
    
    plt.plot(0.7*curv_gaus+left, curv_x, color='k')
    plt.plot([left, m_ori-sd_ori],[m_curv, m_curv], color='k')
    plt.plot([left, m_ori],[m_curv-sd_curv, m_curv-sd_curv], color='0.5')
    plt.plot([left, m_ori],[m_curv+sd_curv, m_curv+sd_curv], color='0.5',)
    ax.text(m_ori/2, m_curv, '$\mu_c$', fontsize=fs, va='top')
    ax.text(m_ori/2.5, m_curv-sd_curv, '$\sigma_c$', fontsize=fs, va='top')

    
else:
    fig = plt.figure(figsize=(8,4))
    ax = plt.subplot(122)
#    for line_seg, a_curv in zip(line_segs, np.roll(curv,to_roll)): 
#        ax.plot([line_seg[0][0], line_seg[1][0]], [line_seg[0][1], line_seg[1][1]], 
#                color=cmap(norm(a_curv)), lw=8)
    ax.plot(x,y, color='k', lw=3)
ax.set_xticks(np.deg2rad(np.array([0, 90, 180, 270])))
directions = ['right', 'up', 'left', 'down' ]
xlabels = [str(ang) for ang, direction in
           zip([0, 90, 180, 270], directions)]
ax.set_xticklabels(xlabels, fontsize=12)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.set_yticks([ 1,  0,  -1])
ax.set_yticklabels([ 'Convex\n+1', '0', '-1\nConcave'], fontsize=12,
                   ha='right')
#plt.scatter(dense_val[match_loc_orig, 1], dense_val[match_loc_orig, 0], color='r')         
#ax.scatter(orig_val[:,1], orig_val[:,0], color='r')
for spine in ['right','top']:
    ax.spines[spine].set_visible(False)
    
ax.plot([left, 2*np.pi],[0, 0], color='0.5')

ax.set_ylim([-1.03,1.03])
ax.set_xlim(left, right)
ax.spines['bottom'].set_bounds(0,right)

ax.tick_params(axis='both', length=0)
ax.set_xlabel('Angular Position ($^\circ$)',fontsize=15)
ax.set_ylabel('Curvature', fontsize=15)
ax.set_aspect(2.5)
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/shape_example.svg', 
            bbox_inches='tight')


if  not schematic_gaussian:
    ax = plt.subplot(121)
    n_pts = np.shape(ashape)[0]
#    line_segs = [[ashape[ind], ashape[ind+1]] for ind in range(n_pts-1)]
#    for line_seg, a_curv in zip(line_segs, curv):    
#        ax.plot([line_seg[0][0], line_seg[1][0]], [line_seg[0][1], line_seg[1][1]], 
#                color=cmap(norm(a_curv)), lw=7)
    ax.plot(ashape[:,0],ashape[:,1],color='k', lw=3)
    for spine in ['bottom','left','right','top']:
        ax.spines[spine].set_visible(False)

#    norm = mpl.colors.Normalize(vmin=0, vmax=1)
#    sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=norm)
#    sm._A = []
#    kw = {'anchor':(1,.5)}
#    norm = mpl.colors.Normalize(vmin=0, vmax=1)
#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm._A = []
#    cbar = plt.colorbar(sm)
    #cbar.set_ticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')
    ax.set_ylim(-2,2)
    ax.set_xlim(-2,2)
    plt.tight_layout()
#cbar.set_label('Curvature', fontsize=15)

    plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/apc_encoding.svg', bbox_inches='tight')
    
#%%
#table of values for example cells.
units = [['Conv2',113], ['Conv2',108], ['Conv2', 126], ['Conv3', 156], ['Conv3', 20],
         ['Conv5', 161], ['Conv5', 144], ['Conv3', 334], ['Conv4', 203], 
         ['FC6', 3030], ['FC7', 3192], ['FC7', 3591], ['FC7', 3639], 
         ['FC8', 271], ['FC8', 433], ['FC8', 722]]

units = [ ['Conv2',108],['Conv2',113], ['Conv2', 126], ['Conv3', 20], ['Conv3', 156],
         ['Conv3', 334],  ['Conv4', 203],['Conv5', 144], ['Conv5', 161], 
         ['FC6', 3030], ['FC7', 3192], ['FC7', 3591], ['FC7', 3639], 
         ['FC8', 271], ['FC8', 433], ['FC8', 722]]

cols = ['apc', 'cur_mean', 'cur_sd', 'or_mean', 'or_sd', 'ti_in_rf']

va = [cnn_an.loc['resp'].loc[unit[0].lower()].iloc[unit[1]][cols] for unit in units]

n = pd.concat(va, 1).T

n['apc'] = n['apc']**0.5
n['or_mean'] = np.rad2deg(n['or_mean']).astype(int)
n['or_sd'] = np.rad2deg(n['or_sd']).astype(int)
decimals = pd.Series([2, 1,2, 0, 0, 2], index=cols)
n = n.round(decimals)
n = pd.concat([pd.DataFrame(units, index=n.index),n],1)
n.columns = ['Layer', 'Unit', 'APC r', r'$\mu_c$', r'$\sigma_c$', r'$\mu_a$', r'$\sigma_a$', 'TI']

template = r'''\documentclass[preview]{{standalone}}
\usepackage{{booktabs}}
\begin{{document}}
{}
\end{{document}}
'''
import six

filename = top_dir + '/analysis/figures/images/v4cnn_cur/table'
with open(filename, 'w') as f:
    f.write(template.format(n.to_latex()))

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    new_data = []
    for row in data.values:
        new_row = []
        for i, item in enumerate(row):
            if i not in [2, 7]:
                new_row.append(item)
            else:
                new_row.append('%1.2f' % item)
        new_data.append(new_row)          
                
    mpl_table = ax.table(cellText=new_data, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

ax = render_mpl_table(n, header_columns=0, col_width=1)
ax.get_figure().savefig(filename+'.pdf')

#%%

tot_num = 0
num_greater = 0
for layer in layers_to_examine:
    a = cnn_an.loc['resp'].drop('v4', level='layer_label')['apc'].loc[layer]**0.5
    tot_num += len(a)
    num_greater += np.sum(a>0.7)


#%%
k = kurtosis(da_0)


<<<<<<< HEAD
#%%

layers_to_examine = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
=======
layers_to_examine = ['relu1','pool1', 'norm1', 'relu2','pool2', 'norm2', 'pool5', 
                     'relu3','relu4','relu5', 'relu6','relu7',]
var = np.concatenate([alt['k_stim'].iloc[layer==all_lays].dropna().ravel() 
                        for layer in layers_to_examine])
n_samps = len(var)
ax.hist(var, bins=n_bins, histtype='step', weights=[1./n_samps,]*n_samps,
         color='m', range=[0,370], log=True,)

layers_to_examine = ['conv1','conv2','conv3','conv4','conv5', 'fc6','fc7','fc8']
all_lays= alt.index.get_level_values(0)
>>>>>>> 4b9ff218dfb66876a9eca6a5220ceffaa6d987fb

layers_to_examine = ['relu1','pool1', 'norm1', 'relu2','pool2', 'norm2', 'pool5', 
                    'relu3','relu4','relu5', 'relu6','relu7',]
b=[apc.loc['resp'].drop('v4', level='layer_label').loc[layer]**0.5
                 for layer in layers_to_examine]
b = [ti_cnn.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine]

for i, a in enumerate(b):
    plt.hist(a[-a.isnull()], histtype='step', cumulative=True, normed=True, 
               bins=100, color=cm.rainbow(np.double(i)/len(b)))
plt.legend(layers_to_examine, loc=2)
plt.xlim(0,1)
plt.xlabel('TI')
plt.savefig('/home/dean/Desktop/sublayers_layers_ti.pdf')
#%%
#import seaborn as sns
layers_to_examine = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
plt.figure(figsize=(2,8))
for i, layer in enumerate(layers_to_examine):
    plt.subplot(len(layers_to_examine),1,i+1)
    x, y = (alt.loc[layer]['apc']**0.5, alt.loc[layer]['ti_in_rf'])
    
    plt.scatter(x,y, s=1)
    plt.xticks([0,0.5,1]);plt.yticks([0,0.5,1]);
    plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
    plt.title(layer+ ' r=' + str(np.round(np.corrcoef(x,y)[0,1],2)))
    plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
plt.gca().set_xticklabels(['0','0.5','1']);plt.gca().set_yticklabels(['0','0.5','1']);

plt.xlabel('APC correlation')
plt.ylabel('TI')
plt.tight_layout()

plt.savefig('/home/dean/Desktop/supp_fig_ti_shape_tuning.jpg')


#%%
import xarray as xr
cnn_name= 'bvlc_reference_caffenetpix_width[ 8.4096606]_x_(34, 194, 21)_y_(34, 194, 21)PC370'

cn = alt.drop('v4', level='layer_label')
ind = (cn['k_pos']>2)&(cn['k_pos']<40)&(cn['k']<40)
#%%
w = 32
rf = [51,99,131,163, 227,227,227]
load_dir = '/loc6tb/'
da = xr.open_dataset(load_dir + 'data/responses/v4cnn/' + cnn_name  + '.nc',
                     chunks={'unit':1000})['resp'].squeeze()

#da = da[...,ind][...,::1]
da = da.transpose('unit', 'y','x',  'shapes')

#da = da[..., 5:-5, 5:-5, :]
'''
for a_rf, layer in zip(rf, layers_to_examine):
    dif = 227-a_rf
    num_steps = dif / 8.
    start= int(round(num_steps/4.))
    print(start)
    if not start==0:
        da[da.coords['layer_label']==layer][:,start:-start,start:-start,:] = 0
    
print(da)
'''
#%%

def norm_cov_unwrap(x):
    #if nxm the get cov mxm
    #print(x.shape)
    x = x.astype(np.float64)
    x = x.reshape((x.shape[0], x.shape[1]**2))
    x = x - np.mean(x, 0, keepdims=True)
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    vnrm = np.linalg.norm(x, axis=0, keepdims=True)
    denominator = np.sum(np.multiply(vnrm.T, vnrm)[diag_inds]) 
    norm_cov = numerator/denominator
    
    return norm_cov



def spearman_correlation(x, dim):
    return xr.apply_ufunc(
        norm_cov_unwrap, x, 
        input_core_dims=[dim],
        dask='parallelized',
        output_dtypes=[float],
        vectorize=[True,True])
    
r = spearman_correlation(da, ['shapes', 'y', 'x']).load()
#%%
layers_to_examine = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
r_sub = r[:]
for layer in layers_to_examine:
    plt.hist(r_sub[r_sub.coords['layer_label'] == layer], 
             cumulative=True, histtype='step', bins=1000, normed=True, range=(-0.1, 1))
plt.xlim(-0.1,1)
plt.legend(layers_to_examine, loc='lower right')

#%%
layers_to_examine = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
plt.figure(figsize=(2,8))
for i, layer in enumerate(layers_to_examine):
    plt.subplot(len(layers_to_examine),1,i+1)
    x, y = (alt.drop('v4', level='layer_label').loc[layer]['apc']**0.5, 
            r[r.coords['layer_label'] == layer])
    
    plt.scatter(x,y, s=1)
    plt.xticks([0,0.5,1]);plt.yticks([0,0.5,1]);
    plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
    plt.title(layer+ ' r=' + str(np.round(np.corrcoef(x,y)[0,1],2)))
    plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
plt.gca().set_xticklabels(['0','0.5','1']);plt.gca().set_yticklabels(['0','0.5','1']);

plt.xlabel('APC correlation')
plt.ylabel('TI')
plt.tight_layout()

plt.savefig('/home/dean/Desktop/supp_fig_ti_shape_tuning.jpg')

#%%
import pickle
with open('/home/dean/Desktop/v4cnn' + '/nets/netwts.p', 'rb') as f:    
    try:
        netwts = pickle.load(f, encoding='latin1')
    except:
        netwts = pickle.load(f)
#%%
def norm_cov(x, subtract_mean=True):

    #if nxm the get cov mxm
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 0, keepdims=True)
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    vnrm = np.linalg.norm(x, axis=0, keepdims=True)
    denominator = np.sum(np.multiply(vnrm.T, vnrm)[diag_inds]) 
    norm_cov = numerator/denominator

    return norm_cov 
data_dir = '/loc6tb/'
net_name = 'bvlc_reference_caffenetpix_width[ 8.4096606]_x_(34, 194, 21)_y_(34, 194, 21)PC370.nc'

da = xr.open_dataset(data_dir + '/data/responses/v4cnn/'+net_name)['resp']

#
wts_by_layer = [layer[1] for layer in netwts]
wtcov = {}
for layer, name in zip(netwts, ['conv1','conv2', 'conv3', 'conv4', 'conv5', 'fc6']):
    a_layer = layer[1]
    temp_wtcov=[]
    if len(a_layer.shape)>2:
        for unit in a_layer:
            unit = unit.reshape((unit.shape[0],) + (np.product(unit.shape[1:]),) )
            temp_wtcov.append(norm_cov(unit))
        wtcov[name] = temp_wtcov
#%%
            
layers_to_examine = ['conv2', 'conv3', 'conv4', 'conv5',]
plt.figure(figsize=(8,2))
for i, layer in enumerate(layers_to_examine):
    plt.subplot(1, len(layers_to_examine),i+1)
    x, y = (wtcov[layer], r[r.coords['layer_label'] == layer], 
            )
    y = y.values
    x = np.array(x)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.scatter(x,y, s=3, facecolor='k', edgecolor='none')
    plt.plot(np.array([0, np.max(x)]), m*np.array([0, np.max(x)]) + c, 'r')

    plt.xticks([0,0.5,1]);plt.yticks([0,0.5,1]);
    plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
    plt.title(layer+ ' r=' + str(np.round(np.corrcoef(x,y)[0,1],2)))
    plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
    if i ==0:
        plt.gca().set_xticklabels(['0','0.5','1']);plt.gca().set_yticklabels(['0','0.5','1']);
        plt.xlabel('Weight Cov')
        plt.ylabel('TI')

plt.tight_layout()
plt.savefig('/home/dean/Desktop/ti_wcov.pdf')
#%%

layers_to_examine = ['conv2', 'conv3', 'conv4', 'conv5']
plt.figure(figsize=(2,8))
for i, layer in enumerate(layers_to_examine):
    plt.subplot(len(layers_to_examine),1,i+1)
    x, y = (alt.drop('v4', level='layer_label').loc[layer]['apc']**0.5,wtcov[layer],  
            
            )
    
    plt.scatter(x,y, s=12)
    plt.xticks([0,0.5,1]);plt.yticks([0,0.5,1]);
    plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
    plt.title(layer+ ' r=' + str(np.round(np.corrcoef(x,y)[0,1],2)))
    plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
plt.gca().set_xticklabels(['0','0.5','1']);plt.gca().set_yticklabels(['0','0.5','1']);

plt.xlabel('Weight Cov')
plt.ylabel('APC')
plt.tight_layout()
plt.savefig('/home/dean/Desktop/wc_apc.jpg')
#%%
layers_to_examine = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
plt.figure(figsize=(2,8))
for i, layer in enumerate(layers_to_examine):
    plt.subplot(len(layers_to_examine),1,i+1)
    x, y = (alt.drop('v4', level='layer_label').loc[layer]['k']**0.1,
            alt.drop('v4', level='layer_label').loc[layer]['apc']**0.1,)
    
    plt.scatter(x,y, s=1)
    #plt.xticks([0,0.5,1]);plt.yticks([0,0.5,1]);
    #plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
    plt.title(layer+ ' r=' + str(np.round(np.corrcoef(x,y)[0,1],2)))
    #plt.gca().set_xticklabels(['','','']);plt.gca().set_yticklabels(['','','']);
#plt.gca().set_xticklabels(['0','0.5','1']);plt.gca().set_yticklabels(['0','0.5','1']);

plt.xlabel('k')
plt.ylabel('apc')
plt.tight_layout()


#%%
import pandas as pd
import seaborn as sns
layer = 'fc8'
data = np.vstack([np.log((alt.drop('v4', level='layer_label').loc[layer]['k']).values),
              (alt.drop('v4', level='layer_label').loc[layer]['apc']**0.5).values,
                                  r[r.coords['layer_label'] == layer].values]).T
da = pd.DataFrame(data, columns = ['k', 'apc', 'ti'])
#pd.scatter_matrix(da)
#sns.lmplot(da)
#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf

results = smf.ols('apc ~ ti + k', data=da).fit()
print(results.summary())
#%%
print(sm.stats.anova_lm(results, typ=2))

fig, ax = plt.subplots()
fig = sm.graphics.plot_regress_exog(results, 1)
