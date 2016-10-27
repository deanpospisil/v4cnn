# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:07:38 2016

@author: dean
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
import d_net_analysis as dn
plt.close('all')


from sklearn.neighbors import KernelDensity

from sklearn.grid_search import GridSearchCV


def d_hist(ax, x, kde=True, hist_alpha=0, bw=None, color='k', bins=100, sample_points=1000):
    x_grid = np.linspace(np.min(x), np.max(x), sample_points)
    
    if bw == None:
        bw = np.std(x)*float(len(x))**(-1/5.) #Silverman Rule-of-Thumb second order normal
        
    kde_skl = KernelDensity(bandwidth=bw)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    est = np.exp(log_pdf)
    ax.plot(x_grid, est, color=color)
    n = ax.hist(x, bins=bins, color=color, histtype='step', 
                             alpha=hist_alpha, lw=1, normed=True)[0]
    return n, est
#ax = plt.subplot(111)
#d_hist(ax, np.random.normal(0,1,100))


    
def beautify(ax=None):
    almost_black = '#262626'
    more_grey = '#929292'
#    text_font = 'serif'
#    number_font = 'helvetica'

    # Get the axes.
    if ax is None:
        #fig = plt.figure(1)
        ax = plt.axes()

    # Remove 'spines' (axis lines)
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    # Make ticks only on the left and bottom (not on the spines that we removed)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    # Now make them go 'out' rather than 'in'
    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, which='both', direction='out', pad=7)
        ax.tick_params(axis=axis, which='major', color=almost_black, length=6)
        ax.tick_params(axis=axis, which='minor', color=more_grey, length=4)

    # Make thinner and off-black
    spines_to_keep = ['bottom', 'left']
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(almost_black)

    # Change the labels & title to the off-black and change their font
    for label in [ax.yaxis.label, ax.xaxis.label, ax.title]:
        label.set_color(almost_black)

    # Change the tick labels' color and font and padding
    for axis in [ax.yaxis, ax.xaxis]:
        # padding
        #axis.labelpad = 20
        # major ticks
        for major_tick in axis.get_major_ticks():
            label = major_tick.label
            label.set_color(almost_black)
        # minor ticks
        for minor_tick in axis.get_minor_ticks():
            label = minor_tick.label
            label.set_color(more_grey)


    # Turn on grid lines for y-only
    #plt.grid(axis='y', color=more_grey)
def small_hist(df, bins, ax, ax_set_range='range_all', sigfig=1, logx=True, 
               logy=False, include_median=False, label='', fontsize=10
               , layers_to_examine=None, bw=None):
    num_colors = len(df.index.levels[0])
    colormap = plt.get_cmap('jet')
    colors = [colormap(1.*i/num_colors) for i in range(num_colors)]  
    colors = ['r','g','b','k', 'm']
    dim2_inds = np.unique(df.index.labels[0])
    dim2_levels = df.index.levels[0]  

    n=[]
    est = []
    for dim2_ind in dim2_inds:            
        var = df.loc[dim2_levels[dim2_ind]].dropna().values
        color = colors[dim2_ind]
        if len(var)>0:
            _ = d_hist(ax, var,  bw=bw, color=color, bins=bins)
            n.append(_[0])
            est.append(_[1])
            #n.append(ax.hist(var, bins=bins, color=color, histtype='step', 
             #                alpha=0.6, lw=2)[0])
    max_n = np.max([np.max(n),np.max(est)])
    the_range = [df.min(), df.max()]
    if logx:
        ax.semilogx(nonposy='clip')
    the_min = np.round(the_range[0], sigfig)
    if np.isclose(the_min, 0):
        the_min = '0'
            
    if ax_set_range == 'symmetric':
        x_plot_bounds = [-np.max(np.abs(the_range)), np.max(np.abs(the_range))]
        spine_bounds = x_plot_bounds
        x_tick_pos = [the_range[0], 0, the_range[1]]
        x_tick_lab = [the_min, ' ', np.round(the_range[1],sigfig)]
    elif ax_set_range=='range_all':          
        x_plot_bounds = [bins[0], bins[-1]]
        spine_bounds = [the_range[0], the_range[1]]
        x_tick_pos = [the_range[0], the_range[1]]
        x_tick_lab = [the_min, np.round(the_range[1], sigfig)]
    if include_median:
        med = np.median(var)
        ax.plot([med, med],[-0.1*max_n, ax.get_ylim()[0]], 
                color=color, clip_on=False)
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_lab, fontsize=fontsize)
    ax.set_xlim(x_plot_bounds[0], x_plot_bounds[1]+x_plot_bounds[1]*.1)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set(lw=0.5)
    ax.spines['bottom'].set_bounds(spine_bounds[0], spine_bounds[1])
    ax.xaxis.set_ticks_position('bottom')
        
    #y axis
    if logy:
        ax.semilogy(nonposy='clip')
        ax.set_ylim(0.5/float(len(var)-1), max_n+max_n*0.1)
    else:
        ax.set_ylim(0, max_n+max_n*0.1)
    ax.set_ylabel(label,
                  rotation='horizontal', labelpad=fontsize*2, 
                  fontsize=fontsize, multialignment='left')
    ax.yaxis.set_label_position('right')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set(lw=0.5)
    ax.yaxis.set_ticks_position('left')
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    for axis in ['x','y']:
        ax.tick_params(axis=axis, which='both', direction='out')
        ax.tick_params(axis, length=0, direction='out', width=0, which='minor')
    ax.spines['left'].set_bounds(1./float(len(var)-1), max_n)
    
    ax.set_yticks([1./float(len(var)-1), max_n,])
    ax.set_yticklabels([' ', np.round(max_n, sigfig+1),], fontsize=fontsize)

def small_mult_hist(df, scale=1, ax_set_range='symmetric', 
                    logx=False, logy=False, bins='auto',
                    include_median=False, sigfig=1, fontsize=12,
                    layers_to_examine=None, bw=None):
    if layers_to_examine == None:
        layers_to_examine = df.index.levels[0]
    m = len(layers_to_examine)
    gs = gridspec.GridSpec(m, 1, width_ratios=[1,],
                            height_ratios=[1,]*m)
    plt.figure(figsize=(4*scale, m*2*scale))
    #fontsize = 10 * scale
    ax_list = [plt.subplot(gs[pos]) for pos in range(m)];


    for dim1, ax in zip(layers_to_examine, ax_list):
        small_hist(df.loc[dim1], bins, ax, label=dim1, 
                   logx=logx,logy=logy, fontsize=fontsize, sigfig=sigfig, bw=bw)

    return ax_list
    
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
    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(ti)),np.arange(len(ti))], names=keys)
    v4pdti  = pd.DataFrame(np.array([ti, k_ti]).T, index=index, columns=['ti_av_cov', 'k'])

    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(apc)),np.arange(len(apc))], names=keys)
    v4pdapc  = pd.DataFrame(np.array([apc.values, k_apc]).T, index=index, columns=['apc', 'k'])
    v4 = pd.concat([v4pdti, v4pdapc])
    return v4

goforit = False
#loading up all needed data
if 'cnn_an' not in locals() or goforit:
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
    v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()
    alt_v4 = process_V4(v4_resp_apc, v4_resp_ti, dmod)

    #shuffle
    v4_resp_apc_null = v4_resp_apc.copy()
    v4_resp_ti_null = v4_resp_ti.copy()

    for  x in range(len(v4_resp_ti_null.coords['x'])):
        for unit in range(len(v4_resp_ti_null.coords['unit'])):
            not_null = ~v4_resp_ti_null[unit,x,:].isnull()
            v4_resp_ti_null[unit, x, not_null] = np.random.permutation(v4_resp_ti[unit, x, not_null].values)
    
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
                        axis=0, 
                        keys=['resp', 's. resp', 'init', 's. wts'], 
                        names=['cond','layer_label','unit'])
    
    cnn_an = cnn_an.swaplevel(i=0, j=1)

fontsize=12
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(top_dir + 'analysis/figures/images/' + 'v4cnn_figures.pdf') as pdf:
    plt.rc('text', usetex=False)

    for layers_to_examine in [['conv1', 'relu1', 'norm1', 'conv2', 'norm2', 'conv5', 'fc6', 'prob', 'v4'],np.array(layer_label).astype(str)]:
        ax_list = small_mult_hist(cnn_an['k'].drop(['s. resp',], level='cond'), 
                                 bins=np.linspace(.99,370,1000), 
                                 logx=True, logy=False, 
                                 fontsize=fontsize, sigfig=1, 
                                 layers_to_examine=layers_to_examine)

        ax_list[0].legend(cnn_an.index.levels[1], frameon=0, fontsize=fontsize)
        ax_list[0].set_title('Kurtosis', fontsize=fontsize)
        plt.tight_layout()
        pdf.savefig()  # or you can pass a Figure object to pdf.savefig
        plt.close()
        
        ax_list = small_mult_hist(cnn_an['apc'][cnn_an['k']<40], 
                                  bins=np.linspace(0,1,20), logx=False, logy=False, 
                                    fontsize=fontsize, sigfig=2, 
                                    layers_to_examine=layers_to_examine, bw=None)
        ax_list[0].legend(cnn_an.index.levels[1], frameon=0, fontsize=fontsize)
        ax_list[0].set_title('APC $R^2$', fontsize=fontsize)
        plt.tight_layout()
        pdf.savefig()  # or you can pass a Figure object to pdf.savefig
        plt.close()
        d_cnn_an = cnn_an['ti_av_cov'][(cnn_an['k']<40)*(cnn_an['k']>2.3)].drop('s. resp', level='cond')
        ax_list = small_mult_hist(d_cnn_an, 
                                  bins=np.linspace(0,1,20), logx=False, logy=False,
                                  fontsize=fontsize, sigfig=2, 

                                  layers_to_examine=layers_to_examine)
        legend_labels = list(cnn_an.index.levels[1][list(np.sort(np.unique(d_cnn_an.index.labels[1])))])
        ax_list[0].legend(legend_labels, frameon=0, fontsize=fontsize)
        ax_list[0].set_title('Normalized Average Covariance', fontsize=fontsize)
        plt.tight_layout()
        pdf.savefig()  # or you can pass a Figure object to pdf.savefig
        plt.close()
