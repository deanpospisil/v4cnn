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

def beautify(ax=None):
    '''Apply beautiful settings to an existing graph (figure 1, specifically).
    FEATURES:
    ----------------------------------------------------------------------------
    General:
    - set all things that were black (axis lines, labels, title, major tick
          labels, major ticks marks) to be off-black (a dark grey)
    Axes:
    - remove the top and right axis lines ('spines')
    - make remaining axis lines thinner
    - make dotted grid for major y lines
    Ticks:
    - turn off only right and top ticks
    - set tick direction to be out (pointing out of graph)
    - set all minor ticks labels and marks to be a lighter grey
    - make major and minor ticks longer
    - make tick numbers farther away (padded) to accomodate longer ticks
    Fonts:
    - set all fonts to be serif (like Times New Roman)
    Things you need to do on your own (that aren't generalizable to put here):
    ----------------------------------------------------------------------------
    - Make sure major and minor ticks are specified correctly. Check out:
          [example] http://matplotlib.org/examples/pylab_examples/
                        major_minor_demo1.html
          [api] http://matplotlib.org/api/ticker_api.html
    - Set plotting color. Use better colors than 'blue'. Check out:
          [example] http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
          [example] http://colorbrewer2.org/
    - Set plotting style. Lines should be very differentiable, with lots of
      visual redundancy. This means the following should be different:
        - markers [e.g. circles, triangles, 'x's, ...]
        - color (see above)
        - and style [e.g. dashed, dotted, line, variations in-between]
    - Set error bar width and cap sizes. Error bars should be thinner than
      plotting lines. Check out:
          [example] http://stackoverflow.com/questions/7601334/how-to-set-the-
                        line-width-of-error-bar-caps-in-matplotlib
    '''
    # Settings
    almost_black = '#262626'
    more_grey = '#929292'
    text_font = 'serif'
    number_font = 'serif'

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

    # To remove the ticks all-together (like in prettyplotlib), do the following
    # instead of tick_left() and tick_bottom()
    #ax.xaxis.set_ticks_position('none')
    #ax.yaxis.set_ticks_position('none')

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
        label.set_family(text_font)

    # Change the tick labels' color and font and padding
    for axis in [ax.yaxis, ax.xaxis]:
        # padding
        #axis.labelpad = 20
        # major ticks
        for major_tick in axis.get_major_ticks():
            label = major_tick.label
            label.set_color(almost_black)
            label.set_family(number_font)
        # minor ticks
        for minor_tick in axis.get_minor_ticks():
            label = minor_tick.label
            label.set_color(more_grey)
            label.set_family(number_font)

    # Turn on grid lines for y-only
    plt.grid(axis='y', color=more_grey)
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
            ax.spines['left'].set_bounds(0.5/float(len(var)-1), max_n_list[pos])
            
            ax.set_yticks([max_n_list[pos],])
            ax.set_yticklabels([np.round(max_n_list[pos], sigfig+1),],fontsize=fontsize)
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

import datetime
from matplotlib.backends.backend_pdf import PdfPages
def kurtosis(da):
    da = da.transpose('shapes', 'unit')
    mu = da.mean('shapes')
    sig = da.reduce(np.nanvar, dim='shapes')
    k = (((da - mu)**4).sum('shapes', skipna=True) / da.shapes.shape[0])/(sig**2)
    return k
    
with PdfPages(top_dir + 'analysis/figures/images/' + 'v4cnn_figures.pdf') as pdf:
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
    hists[0].annotate('%', xy=(-0.1, 0.5), xycoords='axes fraction', 
                        rotation='horizontal', ha='right',va='bottom', 
                        fontsize='x-small', multialignment='right')
    plt.tight_layout()
    d = pdf.infodict()
    d['ModDate'] = datetime.datetime.today()
    pdf.savefig()  # or you can pass a Figure object to pdf.savefig
    plt.close()
#%% 
    #kurtosis
    fn = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p'
    k_apc = kurtosis(v4_resp_apc).values
    try:
        an=pk.load(open(top_dir + 'data/an_results/' + fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(top_dir + 'data/an_results/' + fn,'rb'))
        
    fvx = an[0].sel(concat_dim='r2')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    k_cnn = cnn['k'].dropna()
    layers = cnn.index.get_level_values('layer_label').unique()
    cnns = [[k_cnn.loc[layer].dropna().values.flatten() 
            for layer in layers if layer in layers_to_examine] 
             + [list(k_apc),]
                ,]
    layers = np.array(list(layers) + ['v4',])
    

    hists , n_list = small_mult_hist(cnns, layers_to_examine, scale=0.5, 
                                     ax_set_range='range_all', 
                                     logx=True, logy=False,
                                     bins=np.linspace(0.1,370, 1000),
                                     include_median=False)   
    #%%
    plt.tight_layout()
    pdf.savefig() 
    plt.close()
        #%%
    v4_name = 'V4_362PC2001'
    v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    k_apc = kurtosis(v4_resp_apc).values
    plt.hist(v4_resp_apc[:, np.argmax(k_apc)], bins=30, log=False, normed=False, histtype='step',  range=[0,1])
    plt.hist(v4_resp_apc[:, np.argmin(k_apc)], bins=30,log=False, normed=False, histtype='step', range=[0,1])
    plt.legend(np.round([np.max(k_apc), np.min(k_apc)],1), 
                 title='Kurtosis', loc=1, fontsize='small')
    plt.xlabel('Normalized firing rate');plt.ylabel('Count');plt.xticks([0,1])
    #%%
    pdf.savefig()
    plt.close()
       #%% 
    apc_cnn = cnn['apc'][cnn['k']<np.max(k_apc)].dropna()
    layers = cnn.index.get_level_values('layer_label').unique()
    cnns = [[apc_cnn.loc[layer].dropna().values.flatten() 
            for layer in layers if layer in layers_to_examine] 
             #+ [list(fit_apc),]
                ,]
    #layers = np.array(list(layers) + ['v4',])
    

    hists , n_list = small_mult_hist(cnns, layers_to_examine, scale=0.5, 
                                     ax_set_range='range_all', 
                                     logx=False, logy=False,
                                     bins=np.linspace(0,1,100),
                                     include_median=False,
                                     sigfig=2)   
        #%%
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    #%%
    apc_cnn = cnn['ti_av_cov'][cnn['k']<np.max(k_apc)].dropna()
    layers = cnn.index.get_level_values('layer_label').unique()
    cnns = [[apc_cnn.loc[layer].dropna().values.flatten() 
            for layer in layers if layer in layers_to_examine] 
             #+ [list(fit_apc),]
                ,]
    #layers = np.array(list(layers) + ['v4',])
    

    hists , n_list = small_mult_hist(cnns, layers_to_examine, scale=0.5, 
                                     ax_set_range='range_all', 
                                     logx=False, logy=False,
                                     bins=np.linspace(-1,1,100),
                                     include_median=False,
                                     sigfig=2) 
    plt.tight_layout()
    #%%

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




## Create the PdfPages object to which we will save the pages:
## The with statement makes sure that the PdfPages object is closed properly at
## the end of the block, even if an Exception occurs.
#with PdfPages('multipage_pdf.pdf') as pdf:
#    plt.figure(figsize=(3, 3))
#    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
#    plt.title('Page One')
#    pdf.savefig()  # saves the current figure into a pdf page
#    plt.close()
#
#    plt.rc('text', usetex=True)
#    plt.figure(figsize=(8, 6))
#    x = np.arange(0, 5, 0.1)
#    plt.plot(x, np.sin(x), 'b-')
#    plt.title('Page Two')
#    pdf.attach_note("plot of sin(x)")  # you can add a pdf note to
#                                       # attach metadata to a page
#    pdf.savefig()
#    plt.close()
#
#    plt.rc('text', usetex=False)
#    fig = plt.figure(figsize=(4, 5))
#    plt.plot(x, x*x, 'ko')
#    plt.title('Page Three')
#    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
#    plt.close()
#
#    # We can also set the file's metadata via the PdfPages object:
#    d = pdf.infodict()
#    d['Title'] = 'Multipage PDF Example'
#    d['Author'] = u'Jouni K. Sepp\xe4nen'
#    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
#    d['Keywords'] = 'PdfPages multipage keywords author title subject'
#    d['CreationDate'] = datetime.datetime(2009, 11, 13)
#    d['ModDate'] = datetime.datetime.today()



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