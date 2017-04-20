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
sys.path.insert(0, top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common')
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
        ax.plot([xlim[0], xlim[1]], [0, 0], color=more_grey,lw=2, alpha=0.5)
    if y_line:
        #ax.spines['left'].set_position('center')
        ax.plot([0, 0], [ylim[0], ylim[1]], color=more_grey,lw=2,alpha=0.5)
    
    if unity:
        ax.plot(xlim, xlim, color=more_grey, lw=2,alpha=0.5)

    
        
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
figure_num = [6, 7, 8, 9, 10, 4, 5, 1]
fig_ind = 0
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

goforit = 1
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
                                      prov_commit=False)

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


#%%
plt.hist(alt.loc['conv2']['ti_av_cov'], histtype='step',  bins=20)
plt.hist(init.loc['conv2']['ti_av_cov'], histtype='step', bins=20)
plt.hist(alt.loc['v4']['ti_av_cov'].dropna(), histtype='step', bins=20)
plt.legend(['trained', 'untrained', 'V4'], loc=6, frameon=False)
plt.xlabel('Translation Invariance', fontsize=15)
plt.ylabel('Unit Count', fontsize=15)

beautify(plt.gca())
plt.savefig(top_dir + '/analysis/figures/images/early_layer/TI_conv2', bbox_inches='tight')



#%%
'''
plt.figure(figsize=(10, 10))
ax_dumb = plt.subplot(111)
ax = plt.subplot(111)
resp = v4_resp_apc[:,0]
#m, data = plot_resp_on_shapes(ax, no_blank_image, resp, image_square=20)
#beautify(ax, ['top','right','left','bottom'])
#cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.5)
c_imgs = np.zeros(np.shape(no_blank_image) + (4,))
respsc = (resp - resp.min())
respsc = respsc/respsc.max()

scale = cm.cool(respsc)

for i, a_color in enumerate(scale):
    c_imgs[i, np.nonzero(no_blank_image[i])[0], np.nonzero(no_blank_image[i])[1],:] = a_color

im = ax_dumb.imshow(np.tile(respsc,(2,1)), cmap=cm.cool, interpolation='nearest')

cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.5, ticks=[np.min(respsc), np.mean(respsc), np.max(respsc)])
cbar.ax.set_yticklabels([str(np.round(np.min(respsc).values,1)) 
,'$\mu$',
str(np.round(np.max(respsc).values,1))], fontsize=20) 
cbar.ax.set_ylabel('Normalized\nResponse', rotation='horizontal', fontsize=15, ha='left')

data, im = vis_square(ax, c_imgs)
ax.imshow(data, interpolation='nearest')
beautify(ax, ['top','right','left','bottom'])

plt.savefig(top_dir + '/analysis/figures/images/activity_unsorted.png', bbox_inches='tight')
'''
#%%
'''
plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
resp = v4_resp_apc[:,0]
#m, data = plot_resp_on_shapes(ax, no_blank_image, resp, image_square=20)
#beautify(ax, ['top','right','left','bottom'])
#cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.5)
c_imgs = np.zeros(np.shape(no_blank_image) + (4,))
respsc = (resp - resp.min())
respsc = respsc/respsc.max()

scale = cm.cool(respsc)
resp_sort_inds = np.argsort(resp)[::-1]

for i, a_color in enumerate(scale):
    c_imgs[i, np.nonzero(no_blank_image[i])[0], np.nonzero(no_blank_image[i])[1],:] = a_color

plt.imshow(np.tile(respsc,(2,1)), cmap=cm.cool, interpolation='nearest')
cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.5, ticks=[np.min(respsc), np.mean(respsc), np.max(respsc)])
cbar.ax.set_yticklabels([str(np.round(np.min(respsc).values,1)) 
,'$\mu$',
str(np.round(np.max(respsc).values,1))], fontsize=20) 
cbar.ax.set_ylabel('Normalized\nResponse', rotation='horizontal', fontsize=15, ha='left')

data, im = vis_square(ax, c_imgs[resp_sort_inds])
plt.imshow(data, interpolation='nearest')
beautify(ax, ['top','right','left','bottom'])

plt.savefig(top_dir + '/analysis/figures/images/activity_sorted.png', bbox_inches='tight')
'''
#%%
'''
def plot_resp_on_sort_shapes(ax, shapes, resp, top=25, fs=20, shrink=.5,):
    c_imgs = np.zeros(np.shape(shapes) + (4,))
    respsc = (resp - resp.min())
    respsc = respsc/respsc.max()
    
    scale = cm.cool(respsc)
    resp_sort_inds = np.argsort(resp)[::-1]
    
    for i, a_color in enumerate(scale):
        c_imgs[i, np.nonzero(shapes[i])[0], np.nonzero(shapes[i])[1],:] = a_color
    
    im = ax.imshow(np.tile(respsc,(2,1)), cmap=cm.cool, interpolation='nearest')
    cbar = ax.get_figure().colorbar(im, ax=ax, shrink=shrink, 
            ticks=[np.min(respsc), np.mean(respsc), np.max(respsc)])
    cbar.ax.set_yticklabels([str(np.round(np.min(respsc).values,1)) 
    ,'$\mu$',
    str(np.round(np.max(respsc).values,1))], fontsize=fs) 
    cbar.ax.set_ylabel('Normalized\nResponse', rotation='horizontal', fontsize=fs/1.5, ha='left')
    
    data, im = vis_square(ax, c_imgs[resp_sort_inds][:top])
    ax.imshow(data, interpolation='nearest')
    beautify(ax, ['top','right','left','bottom'])

    
plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
resp = v4_resp_apc[:,0]
plot_resp_on_sort_shapes(ax, no_blank_image, resp, top=25)
ax.set_ylim(500,10)
#m, data = plot_resp_on_shapes(ax, no_blank_image, resp, image_square=20)
#beautify(ax, ['top','right','left','bottom'])
#cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.5)

plt.savefig(top_dir + '/analysis/figures/images/activity_sorted_top.png', bbox_inches='tight')
'''
#%%
'''
plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
resp = v4_resp_apc[:,0]
#m, data = plot_resp_on_shapes(ax, no_blank_image, resp, image_square=20)
#cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.5)
c_imgs = np.zeros(np.shape(no_blank_image) + (4,))
respsc = (resp - resp.min())
respsc = respsc/respsc.max()

scale = cm.cool(respsc)
resp_sort_inds = np.argsort(resp)[::-1]

for i, a_color in enumerate(scale):
    c_imgs[i, np.nonzero(no_blank_image[i])[0], np.nonzero(no_blank_image[i])[1],:] = [0,0,1,1]

#plt.imshow(np.tile(respsc,(5,1)), cmap=cm.cool, interpolation='nearest')
#cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.5, ticks=[np.min(respsc), np.mean(respsc), np.max(respsc)])
#cbar.ax.set_yticklabels([str(np.round(np.min(respsc).values,1)) 
#,'$\mu$',
#str(np.round(np.max(respsc).values,1))], fontsize=20) 
#cbar.ax.set_ylabel('Normalized\nResponse', rotation='horizontal', fontsize=15, ha='left')

data, im = vis_square(ax, c_imgs)
plt.imshow(data, interpolation='nearest')
beautify(ax, ['top','right','left','bottom'])

plt.savefig(top_dir + '/analysis/figures/images/uniform.png', bbox_inches='tight')
'''
#%%
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
    ax.imshow(data, interpolation='nearest')
    #beautify(ax, ['top','right','left','bottom'])
    return data
fs = 9  

plt.figure(figsize=(6*1,5*1))
import matplotlib.gridspec as gridspec
conds = ['resp', 's. resp', 'init. net']
apc = cnn_an['apc'][cnn_an['k']<40]
m = 3
n = 3
gs = gridspec.GridSpec(m, n, width_ratios=[1,.7,.5],
                        height_ratios=[1,]*m, wspace=0)
ax_list = [plt.subplot(gs[pos]) for pos in range(m*n)]

hist_pos = [0,3,6]
hist_dat_leg = []


hist_dat = [[apc.loc[cond].loc['v4'] for cond in ['resp', 's. resp']],]
hist_dat_leg.append({'labels':['Resp.', 'S. Resp.'], 
                     'fontsize':'xx-small','frameon':True,'loc':4 })

hist_dat.append([apc.loc[cond].drop('v4', level='layer_label') for cond in conds])
hist_dat_leg.append({'labels':['Resp.', 'S. Resp.', 'Untrained'], 'fontsize':'xx-small',
                 'frameon':True,'loc':4})



layers_to_examine = ['conv1','conv2','conv3','conv4','conv5', 'fc6','fc7','fc8']
hist_dat.append([apc.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine] + [apc.loc['resp'].loc['v4'],])
hist_dat.append([apc.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine])
hist_dat_leg.append({'title':'CN resp', 'labels':layers_to_examine, 
                    'fontsize':'xx-small' , 'frameon':True, 'loc':4,'markerscale':100})

for leg in hist_dat_leg:
    leg['fontsize'] = fs
    leg['labelspacing'] = 0
for i, ax_ind in enumerate(hist_pos):
    ax = ax_list[ax_ind]
    if not (ax_ind==hist_pos[-1]):
        colors = ['k','g','b','m','c', 'k', '0.5']
    else:
        colors = cm.copper(np.linspace(0,1,len(hist_dat[i])))
        colors[-1] = [1, 0, 0, 0]
        colors = np.array([[226,128,9,1],[190,39,45,1], [127,34, 83,1], [ 119, 93, 153,1], 
                  [54, 58, 100,1], [157,188,88,1], [75,135,71,1],[ 59, 88,62,1], [0,0,0,1]])
        colors = colors / np.array([[255,255,255,1]])
        

    for apc_vals, color in zip(hist_dat[i], colors):
        x = apc_vals.dropna().values
        y_c, bins_c = d_hist(ax, np.sqrt(x), cumulative=True, color=color, alpha=0.75, lw=2)   
    bins_c = np.concatenate([apc_vals.dropna().values for apc_vals in hist_dat[i]]).ravel()
    beautify(ax, spines_to_remove=['top','right'])
    ax.set_xticks([0,0.5,1])
    ax.set_xticklabels([0,0.5,1], fontsize=10)
    ax.spines['left'].set_bounds(0,1)
    ax.set_xlim(0,1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0,  ' ', 1], fontsize=10)
    ax.set_ylim(0,1.1)
    if not (ax_ind==hist_pos[-1]):
        print(ax_ind, hist_pos[-1])
        #leg = ax.legend(**hist_dat_leg[i])
        #plt.setp(leg.get_title(),fontsize=fs)
    else:
        v4 = plt.Line2D((0,1),(0,0), color='r',lw=3)
        early = plt.Line2D((0,1),(0,0), color=cm.copper(0), lw=3)
        late = plt.Line2D((0,1),(0,0), color=cm.copper(1), lw=3)
        #ax.legend([v4, early, late], ['V4','CN Layer 1', 'CN Layer 8'],
        #fontsize=fs , frameon=True, loc=4, labelspacing=0)
    ax.grid()
    
ax_list[0].set_title('Cumulative Distribution', fontsize=12) 
ax_list[0].set_ylabel('Fraction < r', labelpad=0, fontsize=12) 
ax_list[0].text(0.6, 0.1, 'V4', transform=ax_list[0].transAxes)
ax_list[0].text(0.05,0.6, 'Shuffled', transform=ax_list[0].transAxes, color='g', rotation=80)
ax_list[0].text(0.4,0.6, 'Unshuffled', transform=ax_list[0].transAxes, color='k',rotation=45)


ax_list[3].text(0.5, 0.1, 'CNN all layers', transform=ax_list[3].transAxes)
ax_list[3].text(0.2,0.75, 'Untrained', transform=ax_list[3].transAxes, color='b', rotation=60)

ax_list[6].text(0.5, 0.1, 'CNN by layer', transform=ax_list[6].transAxes)
ax_list[6].text(0.22,0.5, 'conv1',transform=ax_list[6].transAxes, color=colors[0])
ax_list[6].text(0.65,0.5, 'fc8', transform=ax_list[6].transAxes, 
                color=colors[-2], fontsize=14)
ax_list[6].text(0.12, 0.09, 'V4', transform=ax_list[6].transAxes, 
                color=colors[-1], fontsize=10)
ax_list[6].set_xlabel('APC fit r', labelpad=0, fontsize=12)


example_cell_inds = [1,4,7]
v4 = cnn_an.loc['resp'].loc['v4']
v4_apc = v4[-v4['apc'].isnull()]
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
                

kw = {'s':2., 'linewidths':0, 'c':'r'}
colorbar = 1
for ax_ind, dat in zip(example_cell_inds, scatter_dat):
    ax = ax_list[ax_ind]
    x,y= scatter_lsq(ax, dat[0].values, dat[1].values, lsq=True,
                     mean_subtract=True, **kw)
    frac_var = np.corrcoef(x.T, y.T)[0,1]
    cartesian_axes(ax, x_line=True, y_line=True, unity=True)
    beautify(ax, spines_to_remove=['top','right', 'left','bottom'])
    ax.set_xticks([]);ax.set_yticks([]);
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y)+min(y)*0.05, max(y)+max(y)*0.05)
    if example_cell_inds[0]==ax_ind:
        #ax.text(0, 0.5, 'Model',
        #                    transform=ax.transAxes, fontsize=fs,
        #                    va='center', ha='right', rotation='vertical')
        ax.set_ylabel('Model',labelpad=0)
        ax.set_xlabel('Unit', labelpad=0)
    params = 'Curv. $(\mu=$' +  str(np.round(dat[1].coords['cur_mean'].values,2))\
    +', $\sigma=$'+ str(np.round(dat[1].coords['cur_sd'].values,2)) + ')'\
    +'\nOri. $(\mu=$'+ str(np.round(np.rad2deg(dat[1].coords['or_mean'].values)))\
    +', $\sigma=$' + str(np.round(np.rad2deg(dat[1].coords['or_sd'].values),0)) + ')' 
    if ax_ind==1:
        ax.set_title('Example units')
        ax.text(0.5, 0.3, '$r=$' +str(np.round(frac_var, 3)), 
                transform=ax.transAxes, fontsize=10)
        ax.text(0.35, 0.1, params, 
                transform=ax.transAxes, fontsize=7, linespacing=0.5)
    else:
        ax.text(0.6, 0.2, '$r=$' +str(np.round(frac_var, 3)), 
                transform=ax.transAxes, fontsize=10)
        ax.text(0.35, -0.1, params, linespacing=.5,
                transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='white',
                                                              ec='none', alpha=0.8))        
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
    ax.text(-0.05, 1.1, label, transform=ax.transAxes,
      fontsize=fs+2, fontweight='bold', va='top', ha='right')
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig'+
            str(figure_num[fig_ind])+ '_apc_figs_v4cnn.pdf', bbox_inches='tight', dpi=500)
fig_ind += 1
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

v4_ex_plt_inds = [1,0]
for plt_ind, ex_cell in zip(v4_ex_plt_inds, v4_ex):
    ax = ax_list[plt_ind]
    ex_cell_resp = ex_cell[1]*(1/.3)
    rf = (ex_cell_resp).mean('shapes')
    corr = np.corrcoef(ex_cell_resp)
    middle = rf.argmax()
    corr_slice=corr[:, middle]
    ax.locator_params(axis='x', nbins=len(rf.values), tight=True);
    
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
    ax.set_xlim(-0.1,len(rf.values)-1+0.1)
    
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
    a = ex_cell_resp.sel(x=1).values
    b = ex_cell_resp.sel(x=0).values
    scatter_lsq(ax, a, b, lsq=0, mean_subtract=0, **{'s':1})
    ax.locator_params(nbins=10, tight=True);
    ax.axis('equal');
    ax.set_xticks(np.round([0, max(a)]), minor=False)
    ax.set_yticks(np.round([0,max(b)]), minor=False)

    beautify(ax);
    fs=12
    ax.set_xlabel('Pos. ' +str(1)+'\nspk/s',labelpad=10, fontsize=fs,va='top',);
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_ylabel('Pos. ' +str(0)+'\nspk/s', rotation=0,labelpad=10, color='k', 
                  ha='right',va='center', fontsize=fs);
    ax.yaxis.set_label_coords(-0.35, 0.35)
    ax.xaxis.set_label_coords(.5, -0.35)
    ax.set_aspect(1)
    ax.plot([-2,166],[-2,166], color='0.5')
    #ax.set_xlim([0, 166])
    #ax.set_ylim([0, 166])
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/v4cnn_cur/fig'
            +str(figure_num[fig_ind])+'_ti_v4.pdf', bbox_inches='tight')
fig_ind += 1
#%%
# just example cells
m = 3
n = 2
plt.figure(figsize=(6,6))
gs = gridspec.GridSpec(m, n, width_ratios=[1,]*n, height_ratios=[1,]*m) 

ax_list = [plt.subplot(gs[pos]) for pos in range(m*n)]
labels = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.']
for ax, label in zip(ax_list, labels):
    ax.text(0, 1.3, label, transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')
rf = open_cnn_analysis(fns[0], layer_label)[1]
cor = open_cnn_analysis(fns[0], layer_label)[0]
av_cors = cor.groupby('layer_label').mean('unit')
av_rfs = rf.groupby('layer_label').mean('unit')

ex_avg_layer = [b'fc7', b'conv5', b'conv2'][::-1]
#ex_avg_layer = [ b'conv1', b'fc7'][::-1]

ex_inds = [1, 3, 5]  

for layer, ex_ind in zip(ex_avg_layer, ex_inds):
    ax = ax_list[ex_ind]
    ax2 = ax.twinx()

    av_cor = av_cors.sel(layer_label=layer)
    av_rf = av_rfs.sel(layer_label=layer)
    av_rf = av_rf/av_rf.max()
    
    ax2.plot(av_rf.coords['x'].values, av_rf.values, alpha=1, 
             lw=2, color='g')
    print(av_rf.values)
    ax.plot(av_cor.coords['x'].values, av_cor.values, alpha=1, lw=2, 
            color='r')

    ax2.set_ylim(0, 1);
    ax.set_ylim(0, 1);
    ax2.set_yticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])

    ax.set_xlim(64, 164)
    ax.set_xticks([64, 114, 164])
    ax.set_xticklabels([])
    ax2.set_yticklabels([])
    ax.set_yticklabels([])
    if ex_ind ==1:
        ax.set_title('Layer Average\n', fontsize=17)

    beautify([ax,ax2], spines_to_remove=['top',], ); 
        
       
ti_cnn = cnn_an[~cnn_an['ti_av_cov'].isnull()]['ti_av_cov'].loc['resp']
ex_cell_inds = [('conv2', 497), ('conv5',3190), ('fc7', 12604),]
ex_cell_tis = [ti_cnn.loc[ind[0]].loc[ind[1]] for ind in ex_cell_inds]
ex_cell_cors = [cor.sel(unit=ind[1]) for ind in ex_cell_inds]
ex_cell_rfs = [rf.sel(unit=ind[1]) for ind in ex_cell_inds]
cn_ex = [cell for cell in zip(ex_cell_tis, ex_cell_cors, ex_cell_rfs, ex_cell_inds)]
ex_cell_inds = [0,2,4]
ti_leg_pos = [[0.97,0.8],[0.97,0.8],[0.97,0.25]]
for ex_cell,  ex_cell_ind, layer, ti_leg in zip(cn_ex,  ex_cell_inds, ['conv2', 'conv5', 'fc7'],ti_leg_pos):
    ax = ax_list[ex_cell_ind]
    ax2 = ax.twinx()
    cell_label = ex_cell[-1]
    cell_ti = ex_cell[0]
    
    ex_cell_rf = ex_cell[2]
    ex_cell_rf /= ex_cell_rf.max()
    ex_cell_cor = ex_cell[1]
    ex_cell_ti = ex_cell[1]    
    ax.plot(ex_cell_rf.coords['x'].values, ex_cell_rf, color='g', lw=2)
    ax2.plot(ex_cell_rf.coords['x'].values, ex_cell_cor, color='r',lw=2, alpha=1)
    ax.set_xlim()
    beautify([ax, ax2], spines_to_remove=['top',]);
    ax.set_ylim(0,1)
    ax2.set_ylim(0,1)
    ax.set_xticks([64,114,164])
    ax.set_xlim([64,164])

    ax2.set_yticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0', ' ', '1'], fontsize=12)

    ax2.set_yticklabels([])
    #ax.text(str(cell_label) + '. TI='+ str(np.round(cell_ti,2)))
    ax.text(-0.38,0.5, layer, color='k', rotation='vertical',
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
ax_list[0].text(0.5,1.01, 'Unit 497', ha='center', va='bottom',
                transform=ax_list[0].transAxes, fontsize=12, fontstyle='italic')
ax_list[0].annotate('', xy=(101, 0.05), xytext=(101, 0.4), ha='center',
            arrowprops=dict(facecolor='black', shrink=0.05),zorder=1, fontsize=8)
ax_list[4].text(.5,1.01, 'Unit 12604', ha='center', va='bottom',
                transform=ax_list[4].transAxes, fontsize=12, fontstyle='italic')
ax_list[2].set_title('Unit 3190', fontstyle='italic')
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig'+str(figure_num[fig_ind])+'_ti_example_and_avg_v4cnn.pdf', bbox_inches='tight')

fig_ind += 1

#%%

plt.figure(figsize=(4,4))
gs = gridspec.GridSpec(2,1, width_ratios=[1,]*1,
                        height_ratios=[1,]*2) 
ax_list = [plt.subplot(gs[pos]) for pos in range(2)]
labels = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.']
for ax, label in zip(ax_list, labels):
    ax.text(-0.1, 1.1, label, transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')
ti_cnn = cnn_an[~cnn_an['ti_av_cov'].isnull()]
ti_cnn = ti_cnn[(ti_cnn['k']>2)&(ti_cnn['k']<40)]['ti_av_cov']

hist_pos = [1,0]
hist_dat_leg = []
hist_dat = [[ti_cnn.loc['resp'].drop('v4', level='layer_label'),
            ti_cnn.loc['init. net'].drop('v4', level='layer_label')]]
hist_dat_leg.append({'labels':['CN', 'CN init.'], 
                     'fontsize':'xx-small','frameon':False,'loc':(-0.2,1) })

layers_to_examine = ['conv1','conv2','conv3','conv4', 'conv5', 'fc6', 'fc7', 'fc8']
#layers_to_examine = ['relu1','relu2','relu3','relu4', 'relu5', 'relu6', 'relu7', 'fc8']
hist_dat.append([ti_cnn.loc['resp'].drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine] + [ti_cnn.loc['resp'].loc['v4'],])
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
        #ax.set_title('Effect of Training on\nTranslation Invariance',fontsize=16)
        colors = ['c', 'b']
        lw=2.5
        ax.text(.25, .70, 'Untrained',
        ha='center', va='bottom', fontstyle='italic',
        transform=ax.transAxes, fontsize=14, color=colors[1])
        ax.text(.6, .5, 'Trained',
        ha='center', va='bottom', fontstyle='italic',
        transform=ax.transAxes, fontsize=14, color=colors[0])
    else:
        #ax.set_title('Translation Invariance\nby Layer',fontsize=16)
        colors = np.array([[226,128,9,1],[190,39,45,1], [127,34, 83,1], [ 119, 93, 153,1], 
                  [54, 58, 100,1], [157,188,88,1], [75,135,71,1],[ 59, 88,62,1], [0,0,0,1]])
        colors = colors / np.array([[255,255,255,1]])
        lw=2
        ax.text(.30, 0.35, 'conv2',ha='center', va='bottom', fontstyle='italic',
        transform=ax.transAxes, fontsize=14, color=colors[1])
        ax.text(.85, 0.35, 'fc8', ha='center', va='bottom', fontstyle='italic',
        transform=ax.transAxes, fontsize=16, color=colors[-2])
        ax.text(.55, 0.25, 'V4',ha='center', va='bottom', fontstyle='normal',
        transform=ax.transAxes, fontsize=14, color=colors[-1])
        #ax.set_title('Translation Invariance',fontsize=16)

        colors = colors     
        lw=2.5

    for ti_val, color in zip(hist_dat[i], colors):
        x = ti_val.dropna().values
        y_c, bins_c = d_hist(ax, x, cumulative=True, color=color, alpha=1,lw=lw)   
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
    ax.set_xticklabels(['0', '0.5', '1'], fontsize=12)

    ax.set_ylim(-0.1,1.001)
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0', ' ', '1'], fontsize=12)
    #ax.set_ylabel('Fraction Units', fontsize=14)
    #leg = ax.legend(**hist_dat_leg[i],ncol=3)
    #plt.setp(leg.get_title(),fontsize=fs)
    ax.set_ylim(bottom=ax.get_ylim()[0] + ax.get_ylim()[0]*0.05, 
            top=ax.get_ylim()[1]+ax.get_ylim()[1]*0.05)
    ax.grid(axis='y')

    #ax.set_xlabel('Translation Invariance', fontsize=14)
ax_list[0].set_xticklabels([])
ax_list[0].set_ylabel('Fraction Units', labelpad=0, fontsize=14)
ax_list[1].set_xlabel('Translation invariance', labelpad=0, fontsize=14)


plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig'
            +str(figure_num[fig_ind])+'_ti_training_and_layer.pdf')
fig_ind += 1

#%%
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
apc_cor = apc**0.5
for layer, color in zip(['conv2','fc7'],[colors[0],colors[-3]]):
    ax.scatter(ti_cnn.loc['resp'].loc[layer][:n_samples], 
               apc_cor.loc['resp'].loc[layer][:n_samples], 
               color=color, s=1, alpha=0.5)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect(1)
beautify(ax)
ax.grid()
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
ax.scatter(x, y, color=colors[-2], marker='*',s=4)

best_v4_ti = ti_cnn.loc['resp'].loc['v4'].max()
best_v4_apc = apc_cor.loc['resp'].loc['v4'].max()
ax.scatter(best_v4_ti, best_v4_apc, color='r', marker='x',s=4)
ax.legend(['conv2', 'fc7', 'Best AN', 'Best V4*'], fontsize=5, loc=3, 
          labelspacing = 0, scatterpoints=1)

#ti_cnn.loc['resp'].drop('v4', level='layer_label').max()

ax = ax_list[1]
layers_to_examine = ['conv2','conv3','conv4', 'conv5', 'fc6', 'fc7', 'fc8']
hist_dat = []
hist_dat = [v4ness.drop('v4', level='layer_label').loc[layer]
                 for layer in layers_to_examine]
for v4ness_val, color in zip(hist_dat, colors):
    x = v4ness_val.dropna().values
    y_c, bins_c = d_hist(ax, x, cumulative=True, color=color, alpha=1, lw=1.1) 
    ax.scatter([np.min(x),], [0,], color=color, marker='|')

ax.set_xlim(1,0)
ax.set_xticks([1,0.5,0])
ax.set_xticklabels(['0', '0.5', '1'][::-1])

ax.set_ylim(1,-0.1)
ax.set_aspect(0.5)
ax.set_yticks([1,0.5,0])
ax.set_yticklabels(['0', ' ', '1'])
beautify(ax)
ax.grid()
ax.set_xlabel('Distance to APC=1 & TI=1', labelpad=2)

nbins = 100
v4_apc_hist, bins = np.histogram(apc_cor.loc['resp'].loc['v4'].dropna().values, 
                           density=True, bins=nbins, range=(0,1))
v4_ti_hist, bins = np.histogram(ti_cnn.loc['resp'].loc['v4'].dropna().values, 
                          density=True, bins=nbins, range=(0,1))
v4_apc_hist /= len(apc_cor.loc['resp'].loc['v4'].dropna().values)
v4_ti_hist /= len(ti_cnn.loc['resp'].loc['v4'].dropna().values)

joint_hist = np.expand_dims(v4_apc_hist,1)[::-1] * np.expand_dims(v4_ti_hist, 0)
#plt.figure()
#plt.imshow(joint_hist)
x, y = np.meshgrid(bins[1:][::-1], bins[1:])
dist = (x**2+y**2)**0.5
dist_sort = np.argsort(dist.ravel())
cum_hist = np.cumsum(joint_hist.ravel()[dist_sort])
cum_hist = cum_hist/max(cum_hist)
end = np.sum((1-best_v4_apc)**2 + (1-best_v4_ti)**2)**0.5
dist_sort_val = dist.ravel()[dist_sort]
plt.plot(dist_sort_val[dist_sort_val>end], cum_hist[dist_sort_val>end], color='r')
ax.scatter([end,], [0,], color=color, marker='|')

ax.text(0.3, 0.65, 'conv2',ha='center', va='bottom', fontstyle='italic',
transform=ax.transAxes, fontsize=14, color='k')
ax.text(.65, 0.35, 'fc8', ha='center', va='bottom', fontstyle='italic',
transform=ax.transAxes, fontsize=16, color=colors[-2])
ax.text(.2, 0.25, 'v4',ha='center', va='bottom', fontstyle='italic',
transform=ax.transAxes, fontsize=14, color='r')
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig'
            +str(figure_num[fig_ind])+'_v4_ness.pdf')

fig_ind += 1

#%%
layers_to_examine = ['conv1', 'relu1', 'norm1',  'conv2', 'fc6', 'prob']
#layers_to_examine = 'all'
name = 'bvlc_reference_caffenetAPC362_pix_width[64.0]_pos_(64.0, 164.0, 51).nc'
cnn = [xr.open_dataset(top_dir + 'data/responses/' + name)['resp'].sel(x=114), ]

name = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(114.0, 114.0, 1)_amp_(100, 255, 2).nc'
cnns = [xr.open_dataset(top_dir + 'data/responses/' + name)['resp'].sel(amp=amp) for amp in [255, 100]] + cnn

name = 'bvlc_reference_caffenet_nat_image_resp_371.nc'
cnn = [xr.open_dataset(top_dir + 'data/responses/' + name)['resp'],]   
cnns = cnns + cnn
cnns = [cnns[0], cnns[-1], cnns[1], cnns[2]]

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
        
for i, a_lay in enumerate(layers_to_examine):
    ax = ax_list[i]

    for a_cnn in cnns:

        all_lays = a_cnn.coords['unit'].layer_label.values.astype(str)
        var = a_cnn[...,all_lays==a_lay].values.ravel()
        n, bins = np.histogram(var, bins=100, normed=False) 
        
        n =  n/float(len(var));
        ax.plot(bins[1:], np.convolve(gaussian(np.linspace(-1,1,20), 0, 0.15), n, mode='same'))
    ax.semilogy(nonposy='clip')
    ax.set_ylim(10/len(var), 1)
    ax.set_xlim(-the_range[i], the_range[i])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([0,])
    ax.set_xticklabels([' ',])
    ax.set_title(a_lay,)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax.set_yticks([1, 1/100])
    ax.set_yticklabels(['',''])
    ax.tick_params('y', length=0, width=0, which='minor')

ax_list[0].legend(['Amp. 255, Width 32 Pix.', 'Photographs', 'Amp. 100, Width 32 Pix.', 
                    'Amp. 255, Width 64 Pix.', ], loc=(-0.3, 1.5), frameon=0, 
                    fontsize=7, markerfirst=False)
ax_list[0].set_ylabel('log(%)')
#ax_list[0].set_xlabel('Response', labelpad=0)
ax_list[0].text(-0.1,-.27, 'Response', transform=ax_list[0].transAxes, color='k', rotation=0)
ax_list[0].set_yticklabels(['1', '.01'])
ax_list[0].set_xticklabels([0,])


plt.tight_layout(h_pad=0.2)
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig'
            +str(figure_num[fig_ind])+'_dynamic_range.pdf',
            bbox_inches='tight')
fig_ind += 1

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
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
k_apc = kurtosis(v4_resp_apc).values

ax = ax_list[0]
ax.set_xlabel('Normalized Firing Rate')
ax.set_ylabel('Fraction Responses', labelpad=0)
ax.set_xticks([0,1])
ax.set_title('Example Response\nDistributions V4')
n_bins = 10
var = v4_resp_apc[:, np.argmax(k_apc)]
n_samps = len(var)
ax.hist(var, bins=n_bins, histtype='step', weights=[1/n_samps,]*n_samps,  range=[0,1],
         color='r', log=True)
ax.hist(v4_resp_apc[:, np.argsort(k_apc)[len(k_apc)//2]], histtype='step', bins=n_bins, 
        range=[0,1], weights=[1/n_samps,]*n_samps, color='b')
ax.hist(v4_resp_apc[:, np.argmin(k_apc)], histtype='step', bins=n_bins, range=[0,1],
        weights=[1/n_samps,]*n_samps, color='g')

#ax.legend(['Max Kurtosis: ' + str(round(np.max(k_apc),1)), 
#           'Median Kurtosis: ' + str(round(np.median(k_apc),1)), 
#           'Min Kurtosis: ' + str(round(np.min(k_apc),1))], loc='upper right', 
#            fontsize=7.6, frameon=False, columnspacing=0)
ax.legend(['Max.', 
           'Median', 
           'Min.'], loc=[0,0], markerfirst=True, 
            fontsize=6.5, frameon=False, columnspacing=0, title='Kurtosis')
ax.tick_params('y', which = 'both', right=0)

ax.set_yticks([1, 0.1, 0.01])
ax.set_yticklabels([1, 0.1, 0.01])
ax.xaxis.set_label_coords(0.5, -0.3)
ax.yaxis.set_label_coords(-0.3, 0.5)

####
ax = ax_list[1]
ax.hist(k_apc, bins=30, histtype='step', weights=[1/len(k_apc),]*len(k_apc), color='k')
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
            arrowprops=dict(ec ='red', facecolor='red', headwidth=6),zorder=1, )
ax.annotate(s='',xy=(3.9,0), xytext=(3.9,0.05),
            arrowprops=dict(ec ='blue', facecolor='blue', headwidth=6),zorder=1, )
ax.annotate(s='',xy=(2.3,0), xytext=(2.3,0.05),
            arrowprops=dict(ec ='green', facecolor='green', headwidth=6),zorder=1,)


#n =  n/float(len(var));
#ax.plot(bins[1:], np.convolve(gaussian(np.linspace(-1,1,20), 0, 0.1), n, mode='same'))
#plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig1_kurtosis_example_v4.pdf')
n_bins = 20


ax = ax_list[2]
k_apc = alt.drop('v4', level='layer_label')['k'].dropna().values
ax.hist(k_apc, bins=n_bins, histtype='step', weights=[1/len(k_apc),]*len(k_apc), 
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
var = np.concatenate([alt['k'].iloc[layer==all_lays].dropna().ravel() 
                        for layer in layers_to_examine])
n_samps = len(var)
ax.hist(var, bins=n_bins, histtype='step', weights=[1/n_samps,]*n_samps,
         color='m', range=[0,370], log=True,)

layers_to_examine = ['conv1','conv2','conv3','conv4','conv5', 'fc6','fc7','fc8']
all_lays= alt.index.get_level_values(0)

var = np.concatenate([alt['k'].iloc[layer==all_lays].dropna().ravel() 
                        for layer in layers_to_examine])
n_samps = len(var)
ax.hist(var, bins=n_bins, histtype='step', weights=[1/n_samps,]*n_samps,
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
            fontsize=7.6, frameon=False, columnspacing=0,borderpad=0)
ax.set_title('Kurtosis Distribution\nCNN')

[[layer] for layer in alt.index.levels[0]]

[beautify(ax) for ax in ax_list]
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig'
            +str(figure_num[fig_ind])+'_kurtosis.pdf')
fig_ind += 1

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
    ax.imshow(filt.T, interpolation = 'nearest')        
    ax.set_xticks([]);ax.set_yticks([]);
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/fig'
            +str(figure_num[fig_ind])+ '_1stfilters.pdf')

#%%