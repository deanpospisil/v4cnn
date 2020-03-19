#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:50:43 2018

@author: dean
"""

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
sys.path.append(top_dir+ 'v4cnn')
sys.path.insert(0, top_dir + 'xarray/');
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
               columns=['apc', ] + coords_to_take + [ 'k_stim',])
    v4 = pd.concat([v4pdti, v4pdapc])
    return v4
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
if sys.platform == 'linux2': 
    data_dir = '/loc6tb/'
else:
    data_dir = top_dir

goforit = False
#loading up all needed data
if 'cnn_an' not in locals() or goforit:
    
    v4_name = 'V4_362PC2001'
    v4_resp_apc = xr.open_dataset(data_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    file = open(data_dir + 'data/responses/v4_apc_109_neural_labels.txt', 'r')
    wyeth_labels = [label.split(' ')[-1] for label in 
                file.read().split('\n') if len(label)>0]
    v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
    fn = data_dir + 'data/models/' + 'apc_models_362.nc'
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp']
    
    apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                      dmod.chunk({}), 
                                      fit_over_dims=None, 
                                      prov_commit=False)

    v4_resp_apc = v4_resp_apc - v4_resp_apc.mean('shapes')
    v4_resp_ti = xr.open_dataset(data_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()
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
    
    cnn_names =['bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',]
    if sys.platform == 'linux2':
        da = xr.open_dataset(data_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
    else:
        da = xr.open_dataset(data_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
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

#%%
labels_file = '/home/dean/caffe/' + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')
labels[146]

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
    
    data = vis_square(ax, c_imgs[resp_sort_inds][:top])
    #ax.imshow(data, interpolation='nearest')
    #beautify(ax, ['top','right','left','bottom'])
    return data
#%%    return data

fs = 9  
figsize= (6,5)
plt.figure(figsize=figsize)
import matplotlib.gridspec as gridspec
conds = ['resp', 's. resp', 'init. net']
apc = cnn_an['apc'][cnn_an['k']<42]
m = 3
n = 3
gs = gridspec.GridSpec(m, n, width_ratios=[1,.7,.5],
                        height_ratios=[1,]*m, wspace=0)
ax_list = [plt.subplot(gs[pos]) for pos in range(m*n)]

hist_pos = [0,3,6]
hist_dat_leg = []

hist_dat = [[cnn_an.loc[cond].loc['v4']['apc'] for cond in ['resp', 's. resp']],]
hist_dat_leg.append({'labels':['Resp.', 'S. Resp.'], 
                     'fontsize':'xx-small', 'frameon':True, 'loc':4 })

hist_dat.append([apc.loc[cond].drop('v4', level='layer_label') for cond in conds])
hist_dat_leg.append({'labels':['Resp.', 'S. Resp.', 'Untrained'], 'fontsize':'xx-small',
                 'frameon':True,'loc':4})

layers_to_examine = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
hist_dat.append([apc.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine] + [cnn_an.loc['resp'].loc['v4']['apc'].dropna(),])
hist_dat.append([apc.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine])
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
        colors.append([1, 0, 0, 0])

    for apc_vals, color in zip(hist_dat[i], colors):
        x = apc_vals.dropna().values
        if i==2:
            lw=1
        else:
            lw=2
        y_c, bins_c = d_hist(ax, np.sqrt(x), cumulative=True, color=color, 
                             alpha=0.75, lw=lw)   
    bins_c = np.concatenate([apc_vals.dropna().values for apc_vals in hist_dat[i]]).ravel()
    beautify(ax, spines_to_remove=['top', 'right'])
    ax.set_xticks([0,0.5,1])
    ax.set_xticklabels([0,0.5,1], fontsize=10)
    ax.spines['left'].set_bounds(0,1)
    ax.set_xlim(0.1,0.81)
    
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
model = v4_apc['models'].iloc[b_unit]
hi_curv_resp = v4_resp_apc.sel(unit=b_unit)
scatter_dat = [[hi_curv_resp, dmod.sel(models=model), 
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
model_resp = dmod.sel(models=model).squeeze()
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
#%%
example_units = [('conv2', 113), ('conv2', 108), ('conv2', 126), ('conv3', 156), ('conv3', 20),
 ('conv5', 161), ('conv5', 144), ('conv3', 334), ('conv4', 203), ('fc6', 3030), 
 ('fc7', 3192), ('fc7', 3591), ('fc7', 3639), ('fc8', 271), ('fc8', 433), ('fc8', 722)]
layer_label = da_0.coords['layer_label'].values
layer_unit = da_0.coords['layer_unit'].values
n_plots= len(example_units)
plt.figure(figsize=(3,20))

for i, unit in enumerate(example_units):
    ind = (layer_unit == unit[1])*(layer_label==unit[0])
    unit_resp = da_0[:, ind]
    unit_resp = unit_resp.reindex_like(model_resp)
    data = plot_resp_on_sort_shapes(ax, no_blank_image, np.squeeze(unit_resp.values), top=16, fs=fs, 
                                    shrink=0.75, colorbar=colorbar)
    plt.subplot(n_plots, 1, i+1)
    plt.imshow(data[...,0])
    plt.title(unit)
    plt.xticks([]);plt.yticks([])
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'+
            'max_resp_examples.pdf', bbox_inches='tight',
            dpi=500)

#%%
plt.figure(figsize=(3,20))
for i, unit in enumerate(example_units):
    ind = (layer_unit == unit[1])*(layer_label==unit[0])
    unit_resp = da_0[:, ind]
    unit_resp = unit_resp.reindex_like(model_resp)
    data = plot_resp_on_sort_shapes(ax, no_blank_image[::-1], np.squeeze(unit_resp.values[::-1]), top=16, fs=fs, 
                                    shrink=0.75, colorbar=colorbar)
    plt.subplot(n_plots, 1, i+1)
    plt.imshow(data[...,0])
    plt.title(unit)
    plt.xticks([]);plt.yticks([])
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'+
            'min_resp_examples.pdf', bbox_inches='tight',
            dpi=500)


