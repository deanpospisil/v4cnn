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
import d_net_analysis as dn

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
def small_mult_hist(x, labels, scale=1, ax_set_range='symmetric', 
                    logx=False, logy=False, bins='auto',
                    include_median=False, sigfig=1):
    m = len(labels)
    num_colors = len(x)
    colormap = plt.get_cmap('jet')
    colors = [colormap(1.*i/num_colors) for i in range(num_colors)]
        
    gs = gridspec.GridSpec(m, 1, width_ratios=[1,],
                            height_ratios=[1,]*m)
    plt.figure(figsize=(5*scale, m*2*scale))
    
    fontsize = 10 * scale
    
    y_hists = [];n_list = []

    max_list = np.zeros((m, len(x)))
    min_list = np.zeros((m, len(x)))
    max_n_list= np.zeros((m, len(x)))
    for i_an_x, an_x in enumerate(x):
        for x_col, pos in zip(an_x, range(m)):
            var = np.array(x_col)
            max_list[pos, i_an_x] = np.max(var)
            min_list[pos, i_an_x] = np.min(var)
            n, _bins = np.histogram(var, bins=bins, normed=False)
            n =  n/float(len(var))
            max_n_list[pos, i_an_x] = np.max(n)
            
    max_list = np.max(max_list, 1)
    min_list = np.min(min_list, 1)
    max_n_list = np.max(max_n_list, 1)

    for color, an_x in zip(colors, x):
        for x_col, pos in zip(an_x, range(m)):

            ax = plt.subplot(gs[pos])
            var = np.array(x_col)
            the_range = (min_list[pos], max_list[pos])
            
            if type(bins)==int:
                _bins = bins
            elif type(bins)==type(list()) or type(bins)==type(np.array([])):
                beg = np.searchsorted(bins, min_list[pos])
                fin = np.searchsorted(bins, max_list[pos])
                if beg==0 or fin>=len(bins):
                    print('the bins provided do not contain the data')
                _bins = bins[beg-1:fin+1] 

            n, _bins = np.histogram(var, bins=_bins, normed=False)
            n =  n/float(len(var))
            n = [0,] + list(n) + [0,]
            _bins = [_bins[0],] + list(_bins)
            ax.step(_bins, n, where='post', lw=0.5, alpha=0.7, color=color)
            naked_plot([ax,])
            #x axis
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
                x_plot_bounds = [min(min_list), max(max_list)+max(max_list)*0.1]
                spine_bounds = [the_range[0], the_range[1]]
                x_tick_pos = [the_range[0], the_range[1]]
                x_tick_lab = [the_min, np.round(the_range[1],sigfig)]
            if include_median:
                med = np.median(var)
                ax.plot([med, med],[-0.1*max_n_list[pos], ax.get_ylim()[0]], 
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
                ax.set_ylim(0.5/float(len(var)-1), max_n_list[pos]+max_n_list[pos]*0.1)
            else:
                ax.set_ylim(0, max_n_list[pos]+max_n_list[pos]*0.1)
            ax.set_ylabel(str(labels[pos]) +'\n n = '+str(int(len(var))),
                          rotation='horizontal', labelpad=fontsize*2, 
                          fontsize=fontsize, multialignment='left')
            ax.yaxis.set_label_position('right')
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set(lw=0.5)
            ax.yaxis.set_ticks_position('left')
            for axis in ['x','y']:
                ax.tick_params(axis=axis, which='both', direction='out')
                ax.tick_params(axis, length=0, direction='out', width=0, which='minor')
            ax.spines['left'].set_bounds(1./float(len(var)-1), max_n_list[pos])
            
            ax.set_yticks([1./float(len(var)-1), max_n_list[pos],])
            ax.set_yticklabels([' ', np.round(max_n_list[pos], sigfig+1),], fontsize=fontsize)
            #beautify(ax)
            
            y_hists.append(ax) 
            n_list.append(n)
 
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
    fontsize = 8
    y_hists = []
    for y_col, pos in zip(y, y_hist_pos):
        _=plt.subplot(gs[pos])
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


def response_distribution_over_layers(cnns, layers_to_examine='all'):
    cnn_val_lists = []
    for cnn in cnns:
        cnn = cnn.squeeze()
        cnn = cnn.transpose('unit', 'shapes')
        try:
            cnn = cnn.drop(-1,'shapes')
        except:
            cnn = cnn.drop(0,'shapes')
            
        all_lays = cnn.coords['unit'].layer_label.values.astype(str)
        if layers_to_examine == 'all':
            unique_inds = np.unique(all_lays, return_index=True)[1]
            layers = [all_lays[ind] for ind in np.sort(unique_inds)]
        else:
            layers = layers_to_examine
        
        cnn_val_lists.append([cnn[a_layer==all_lays,].values.flatten() 
                    for a_layer in layers])
    hists , n_list = small_mult_hist(cnn_val_lists, layers_to_examine, 
                                     scale=0.75, bins=300, logy=True)   
    return hists     

def kurtosis(da):
    da = da.transpose('shapes', 'unit')
    mu = da.mean('shapes')
    sig = da.reduce(np.nanvar, dim='shapes')
    k = (((da - mu)**4).sum('shapes', skipna=True) / da.shapes.shape[0])/(sig**2)
    return k

def open_cnn_analysis(fn):
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
    apc = dn.ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), dmod.chunk({}), fit_over_dims=None, prov_commit=False)
    k_apc = list(dn.kurtosis(v4_resp_apc).values)
    k_ti = list(dn.kurtosis(v4_resp_ti.mean('x')).values)

    keys = ['layer_label', 'unit']
    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(ti)),np.arange(len(ti))], names=keys)
    v4pdti  = pd.DataFrame(np.array([ti, k_ti]).T, index=index, columns=['ti_av_cov', 'k'])

    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(apc)),np.arange(len(apc))], names=keys)
    v4pdapc  = pd.DataFrame(np.array([apc.values, k_apc]).T, index=index, columns=['apc', 'k'])
    v4 = pd.concat([v4pdti, v4pdapc])
    return v4

goforit = True
#loading up all needed data
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
    rf = None
    da = v4_resp_ti.transpose('unit', 'x', 'shapes')
    
    fns = [
    'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_null_analysis.p'
    ]
    
    alt = pd.concat([open_cnn_analysis(fns[0])[-1], alt_v4], axis=0)
    init = open_cnn_analysis(fns[1])[-1]
    shuf = open_cnn_analysis(fns[2])[-1]
    null = pd.concat([open_cnn_analysis(fns[3])[-1], null_v4], axis=0)
    cnn_an = pd.concat([alt, null, init, shuf ], 
              axis=0, keys=['alt','null', 'init', 'shuf'])
    

    v4_resp_apc_pd = v4_resp_apc[:, apc_fit_v4.argsort().values].to_pandas()
    
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
    fit_best_mods_pd = pd.DataFrame(np.array(fit_best_mods_pd).squeeze().T)
                                    #columns=np.round(np.sort(apc_fit_v4.values),3))

 #%%    
import datetime
from matplotlib.backends.backend_pdf import PdfPages

    
with PdfPages(top_dir + 'analysis/figures/images/' + 'v4cnn_figures_old.pdf') as pdf:
    #dynamic range
    plt.rc('text', usetex=False)
    layers_to_examine = ['conv1', 'relu1', 'norm1', 'conv2', 'norm2', 'conv5', 'fc6', 'prob']
    
    name = 'bvlc_reference_caffenetAPC362_pix_width[64.0]_pos_(64.0, 164.0, 51).nc'
    cnn = [xr.open_dataset(top_dir + 'data/responses/' + name)['resp'].sel(x=114), ]
    
    name = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(114.0, 114.0, 1)_amp_(100, 255, 2).nc'
    cnns = [xr.open_dataset(top_dir + 'data/responses/' + name)['resp'].sel(amp=amp) for amp in [100, 255]] + cnn
    
    name = 'bvlc_reference_caffenet_nat_image_resp_371.nc'
    cnn = [xr.open_dataset(top_dir + 'data/responses/' + name)['resp'],]   
    cnns = cnns + cnn
           
    hists = response_distribution_over_layers(cnns, layers_to_examine)
    
    hists[0].legend(['100 (Amp.)', '255 (Amp.)', '64 (pix)', 'Nat.'], frameon=0, fontsize='xx-small')
    hists[0].set_xlabel('Response')
    hists[0].annotate('%', xy=(-1.1 , 0.5), xycoords='axes fraction', 
                        rotation='horizontal', ha='right',va='bottom', 
                        fontsize='x-small', multialignment='right')
    plt.tight_layout()
    d = pdf.infodict()
    d['ModDate'] = datetime.datetime.today()
    pdf.savefig()  # or you can pass a Figure object to pdf.savefig
    plt.close()

    v4_name = 'V4_362PC2001'
    v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    k_apc = kurtosis(v4_resp_apc).values
    plt.hist(v4_resp_apc[:, np.argmax(k_apc)], bins=30, log=False, normed=False, histtype='step',  range=[0,1])
    plt.hist(v4_resp_apc[:, np.argmin(k_apc)], bins=30,log=False, normed=False, histtype='step', range=[0,1])
    plt.legend(np.round([np.max(k_apc), np.min(k_apc)],1), 
                 title='Kurtosis', loc=1, fontsize='small')
    plt.xlabel('Normalized firing rate');plt.ylabel('Count');plt.xticks([0,1])
    plt.title('Example V4 Response Histogram')
    pdf.savefig()
    plt.close()

'''                 
                      
    n_subplot = len(layers)
    for i, layer in enumerate(layers):
        plt.subplot(n_subplot, 1, i+1)
        vals = cnn.loc[layer].dropna().values.flatten()
        hist, bins = np.histogram(vals, bins=100, normed=False)
        hist =  hist/float(len(vals))
        hist = [0,] + list(hist)
        plt.step(bins, hist)
        plt.xlim(min(cnn),400)
    
        plt.gca().set_ylabel(layer, ha='right', rotation=0, labelpad=25)
        plt.gca().yaxis.set_label_position("right")
        plt.xscale('log', nonposy='clip');
        plt.yscale('log', nonposy='clip');
        xticks = np.round(np.array([np.min(k_cnn), np.max(k_cnn)]),1)
        plt.xticks(xticks, xticks);
        plt.plot([max(k_apc),]*2,[0,max(hist)], lw=2, color='green')
        plt.plot([np.median(vals),]*2,[0,max(hist)], lw=2, color='red')
    

    #plt.savefig(top_dir + 'analysis/figures/images/' + '100_255_amp_response_dist.pdf')


This is a demo of creating a pdf file with several pages,
as well as adding metadata and annotations to pdf files.



from collections import Counter

id_layer=cnn['prob'==all_lays,]
shape_id = id_layer.argmax('unit')
counts = Counter(shape_id.values)

ids = np.array([key for key in counts.keys()])
number = np.array([n for n in counts.values()])
ids = ids[np.argsort(number)][::-1]
number = np.sort(number)[::-1]

labels_file = top_dir + 'data/image_net/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')
print(labels[ids])
print(number/370.)
'''