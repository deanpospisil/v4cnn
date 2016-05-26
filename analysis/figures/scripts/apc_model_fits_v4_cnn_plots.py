# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:23:36 2016

plots of v4 APC params

@author: dean
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')


import d_img_process as imp
import xarray as xr

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=0, padval=0):
    plt.figure(figsize = (10,7.8))
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.xticks([])
    plt.yticks([])
    plt.imshow(data, interpolation='bicubic', cmap = cm.Greys_r)
    cbar=plt.colorbar(shrink=0.8)
    cbar.ax.set_ylabel('Normalized Firing Rate', rotation= 270, labelpad=15, fontsize = 15,)
    cbar.ax.yaxis.set_ticks([0,.25,.5,.75, 1])
    cbar.ax.set_yticklabels(['0', .25, .5, .75, 1])
    cbar.solids.set_rasterized(True)
    plt.tight_layout()
    plt.show()
    return data


def tick_format_d_int(x, pos):
    if x==0:
        return('0')
    else:
        return(str(round(x,0)).split('.')[0])

def tick_format_d(x,pos,dec=2):
    if x==0:
        return('0')
    else:
        return(round(x,dec))
#def nice_axes(axes):
#    for i, an_axes in enumerate(axes):
#        an_axes.xaxis.set_tick_params(length=0)
#        an_axes.yaxis.set_tick_params(length=0)
#        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
#        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
def nice_axes(axes, xticks=None, yticks=None, nxticks=2, nyticks=2):
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
        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d_int))
        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))

def find_count_unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx, u_counts = np.unique(b, return_index=True, return_counts=True)
    unique = a[idx]

    return unique, u_counts

def scatter_w_marginals(x, y, titlex, titley, xlim, ylim, xbins=None, ybins=None,
                        title=None):

    #first check if there is overlap in x, y
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1] )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[3])

    a = np.vstack((x,y)).T
    unique, counts = find_count_unique_rows(a)
    u_count = np.sort(np.unique(counts))

    if np.max(u_count)>1:#check there are in fact overlapping  points
        counts_s = (counts/np.double(max(counts))*100.)
        ax2.scatter(unique[:,0], unique[:,1], s=counts_s)

        u_count_s = np.sort(np.unique(counts_s))

        if len(u_count)>2:
            size = [np.min(u_count_s), np.median(u_count_s), np.max(u_count_s)]
            cnt = [np.min(u_count), int(np.median(u_count)), np.max(u_count)]
        else:
            size = [np.min(u_count_s),  np.max(u_count_s)]
            cnt = [np.min(u_count), np.max(u_count)]

        for_legend={'plt':[], 'label':[]}
        [for_legend['plt'].append(plt.scatter([],[], s=a_size)) for a_size in size]
        [for_legend['label'].append(str(a_cnt)) for a_cnt in cnt]

        # Put a legend to the right of the current axis
        ax2.legend(for_legend['plt'], for_legend['label'], frameon=True,
                         fontsize='medium', loc = 'top left', handletextpad=1,
                         title='Counts', scatterpoints = 1,fancybox=True,
                         framealpha=0.5, bbox_to_anchor=(-.15, -0.1))
    else:
        ax2.scatter(x, y)

    ax1.hist(y, orientation='horizontal', bins=ybins, range=ylim, align='mid')
    ax3.hist(x, orientation='vertical', bins=xbins, range=xlim, align='mid')

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_title(title)
    ax2.xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax2.yaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])

    ax1.xaxis.set_major_locator(mtick.LinearLocator(numticks=2, presets=None))
    ax1.yaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax1.yaxis.set_label_text(titley)

    ax3.xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax3.yaxis.set_major_locator(mtick.LinearLocator(numticks=2, presets=None))
    ax3.xaxis.set_label_text(titlex)
    ax3.set_xlim(xlim)
    ax1.set_ylim(ylim)
    #nice_axes(fig.axes)
    plt.show()
    return fig

def correct_bins_for_hist(bins):
    dif = np.diff(bins)
    bin_inds = range(len(bins)) + [len(bins)-1,]
    dif_inds = [0,] + range(len(dif)) + [len(dif)-1,]
    difs_to_add = dif[dif_inds]
    difs_to_add[:-1] = -difs_to_add[:-1]
    bins = bins[bin_inds]
    bins = bins + difs_to_add/2.

    return bins

import pickle
#plt.subplot(221)
#with open(top_dir + 'data/models/degen_16x16.p', 'rb') as f:
#    ds_list = pickle.load(f)
#
#ds_list[0]['real']
#plt.hist(ds_list[0]['real'].values, alpha=0.5, range=[0,1],bins=40,color='red')
#plt.hist(ds_list[0]['shuf'].values, alpha=0.5, range=[0,1],bins=40,color='blue')
#plt.title('degen_16x16')
#
#plt.subplot(222)
#
#with open(top_dir + 'data/models/no_degen_16x16.p', 'rb') as f:
#    ds_list = pickle.load(f)
#
#plt.hist(ds_list[0]['real'].values, alpha=0.5, range=[0,1],bins=40,color='red')
#plt.hist(ds_list[0]['shuf'].values, alpha=0.5, range=[0,1],bins=40,color='blue')
#
#plt.title('no_degen_16x16')
#
#
#plt.subplot(223)
#
#with open(top_dir + 'data/models/degen_16x10.p', 'rb') as f:
#    ds_list = pickle.load(f)
#
#plt.hist(ds_list[0]['real'].values, alpha=0.5, range=[0,1],bins=40,color='red')
#plt.hist(ds_list[0]['shuf'].values, alpha=0.5, range=[0,1],bins=40,color='blue')
#
#plt.title('degen_10x10')
#
#plt.subplot(224)
#with open(top_dir + 'data/models/no_degen_16x10.p', 'rb') as f:
#    ds_list = pickle.load(f)
#
#plt.hist(ds_list[0]['real'].values, alpha=0.5, range=[0,1],bins=40,color='red')
#plt.hist(ds_list[0]['shuf'].values, alpha=0.5, range=[0,1],bins=40,color='blue')
#
#plt.title('no_degen_10x10')
#
#a=plt.gcf().axes
#for ax in a:
#    ax.set_ylim(0,5000)

#BRUTE FORCE FITS
#V4
plt.close('all')

fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models': 100, 'shapes': 370}  )['resp']

with open(top_dir + 'data/models/ds_list_no_degen.p', 'rb') as f:
    ds_list= pickle.load(f)

da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
daa=daa.loc[:, 0, :]#without translation
daa = daa.isel(shapes=da.coords['shapes'])


#plt.close('all')
fig = plt.figure()
nbins=30
n_resp_sets = len(ds_list)
for i, cor in enumerate(ds_list):
    ax = plt.subplot(n_resp_sets, 1, i+1)

    ax.hist(cor['real'].values, bins=nbins, range= [0,1])
    n, bins, patches = ax.hist(cor['shuf'].values, bins=nbins, range= [0,1], alpha=0.7)
    plt.ylim(0,max(n)+max(n)*0.05)
    if i+1 == n_resp_sets:
        ax.xaxis.set_label_text('Correlation Coefficient', fontsize='x-large')
    else:
        plt.legend([ 'Original', 'Shuffled'])
    ax.yaxis.set_label_text('Unit Count', fontsize='x-large')
    plt.title(cor['real'].attrs['type'])
    #ax.tight_layout()


nice_axes(plt.gcf().axes, nxticks=5, nyticks=5)
plt.tight_layout()
plt.show()
plt.savefig(top_dir + 'analysis/figures/images/real_shuf_apc_fits_cnn_v4.eps')
plt.savefig(top_dir + 'analysis/figures/images/real_shuf_apc_fits_cnn_v4.png')

def plot_resp_on_shapes(imgStack, resp, description, image_square = 19):
    resp_sc = (resp.values*0.8 +.2)
    imgStack = imgStack*resp_sc.reshape(362,1,1)
    #sort images
    sortStack = imgStack[list(reversed(np.argsort(resp_sc))),:,:]
    sortStack = np.array([imp.centeredCrop(img, 64, 64) for img in sortStack])
    data = vis_square(sortStack[0:image_square**2])
    plt.title('Ranked response. ' + description, fontsize='x-large')
    plt.tight_layout()
    plt.show()

#responses plotted on shapes for best fits
baseImageDir = top_dir +'images/baseimgs/PCunique/'
files = os.listdir(baseImageDir)
imgStack = np.zeros(( 362, 227, 227))
for f in files:
    if '.npy' in f:
        num, ext = f.split('.')
        num = int(num)
        imgStack[num, :,: ] = np.load(baseImageDir + f)

alex=ds_list[0]['real']
bestunit = alex[alex.argsort().values[-1]]
resp = daa.isel(unit=int(bestunit.coords['unit'].values))
description = 'Layer: '+ str(resp.coords['layer_label'].values) +', unit: ' + \
str(resp.coords['layer_unit'].values) + ', APC-fit: ' + str(round(bestunit.values,2))
plot_resp_on_shapes(imgStack, resp, description, image_square = 10)
plt.savefig(top_dir + 'analysis/figures/images/example_cell_rank_ordered_shapes.png')

v4=ds_list[1]['real']
bestunit = v4[v4.argsort().values[-1]]
resp = da.isel(unit=int(bestunit.coords['unit'].values))
description = 'V4, Unit: ' +  str(resp.coords['unit'].values) + ', APC-fit: ' + str(round(bestunit.values,2))
plot_resp_on_shapes(imgStack, resp, description, image_square = 10)
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/alex_example_cell_rank_ordered_shapes.png')

#make the scatter plots.
plt.figure()
rthresh=0.55

ybins_cur = correct_bins_for_hist(np.unique(dmod.coords['cur_sd']))
xbins_cur = correct_bins_for_hist(np.unique(dmod.coords['cur_mean']))

ybins_or = correct_bins_for_hist(np.unique(dmod.coords['or_sd'].values/(2*np.pi))*360)
xbins_or = correct_bins_for_hist((((np.unique(dmod.coords['or_mean'].values)/(2*np.pi))*360)%360))

cor = v4.copy()
threshFits = cor[cor>rthresh]

tot = len(cor)
ncell = len(threshFits['unit'])
title = 'r > ' + str(rthresh) + ', ' + str(ncell) + '/' + str(tot)

threshFits.coords['or_mean'] = ((threshFits.coords['or_mean']/(2*np.pi))*360)%360
threshFits.coords['or_sd'] = ((threshFits.coords['or_sd']/(2*np.pi))*360)

fig = scatter_w_marginals(threshFits.coords['or_mean'].values, threshFits.coords['or_sd'].values,
                    'Mean Orientation', 'SD Orientation',
                    xlim= [0, 360],
                    #ylim= [0, np.round(np.max(threshFits['or_sd'] )+1,-1)],
                    ylim = [0, 180],
                    title = title,
                    ybins=ybins_or,
                    xbins=xbins_or)
plt.show()
plt.savefig(top_dir + 'analysis/figures/images/v4_apc_ori_brute.png')

fig = scatter_w_marginals(threshFits.coords['cur_mean'].values, threshFits.coords['cur_sd'].values,
                    'Mean Curve', 'SD Curve',
                    xlim= [-1, 1],
                    ylim= [0, np.round(np.max(threshFits['cur_sd']),0).values],
                    title = title,
                    ybins=ybins_cur,
                    xbins=xbins_cur)
plt.savefig(top_dir + 'analysis/figures/images/v4_apc_curv_brute.png')


cor = alex.copy()
rthresh=0.7
threshFits = cor[cor>rthresh]
threshFits.coords['or_mean'] = ((threshFits.coords['or_mean'].values/(2*np.pi))*360)%360
threshFits.coords['or_sd'] = ((threshFits.coords['or_sd'].values/(2*np.pi))*360)

tot = len(cor)
ncell = len(threshFits['unit'])
title = 'r > ' + str(rthresh) + ', ' +str(ncell) + '/'+ str(tot)

ncell = len(threshFits['unit'])
fig = scatter_w_marginals(threshFits.coords['or_mean'].values, threshFits.coords['or_sd'].values,
                    'Mean Orientation', 'SD Orientation',
                    xlim= [0, 360],
                    ylim = [0, 180],
                    title = title, ybins=ybins_or, xbins=xbins_or)
plt.savefig(top_dir + 'analysis/figures/images/alex_apc_ori_brute.png')

fig = scatter_w_marginals(threshFits.coords['cur_mean'].values,
                        threshFits.coords['cur_sd'].values,
                        'Mean Curve', 'SD Curve',
                        xlim= [-1, 1],
                        ylim= [0, np.round(np.max(threshFits['cur_sd']),0).values],
                        title=title, ybins=ybins_cur, xbins=xbins_cur)

plt.savefig(top_dir + 'analysis/figures/images/alex_apc_curv_brute.png')

import scipy.stats.mstats as ms
m = ms.mode(cor.coords['models'].values)
dmod.models
com = daa.sel(unit=cor.isel(unit=(cor.models==m[0][0])).unit)
com = com - com.mean('shapes')
com = com/((com**2).sum('shapes'))**0.5
com = com.values

plt.figure()

resp =dmod.sel(models=int(m[0][0]))
plot_resp_on_shapes(imgStack, resp, 'most common model', image_square = 19)
plt.figure()
plt.plot(com)
plt.figure()
com = daa.sel(unit=cor.isel(unit=(cor.models==m[0][0])).unit)[:,22]
com = com - com.mean('shapes')
com = com/((com**2).sum('shapes'))**0.5
resp = com
plot_resp_on_shapes(imgStack, resp, 'actual resp', image_square = 19)


vals, counts = np.unique(cor.coords['models'].values, return_counts=1 )
threshFits.coords['cur_mean'].values
threshFits.coords['cur_sd'].values
x=threshFits.coords['or_mean'].values
y=threshFits.coords['or_sd'].values

a = np.vstack((x,y)).T
unique, counts = find_count_unique_rows(a)
u_count = np.sort(np.unique(counts))

'''
#NON-LIN FITS
#correlation histogram
plt.figure()
v4fits = xr.open_dataset(top_dir + 'data/an_results/V4_370PC2001_LSQnonlin.nc')['fit']
v4fits = v4fits.to_pandas()
h = plt.hist(v4fits['r'].values)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Cell Count')
plt.xticks([0,0.25,0.5,0.75,1])
plt.xlim([0,1])
plt.ylim([0, 20])
plt.tight_layout()
nice_axes(plt.gcf().axes)
plt.savefig(top_dir + 'analysis/figures/images/v4_apc_correlation_hist.png')

#fit parameters
rthresh = 0.55
v4fits['mori'] = ((v4fits['mori']/(2*np.pi))*360)%360
v4fits['sdori'] = ((v4fits['sdori']/(2*np.pi))*360)
k = v4fits.keys()
threshFits = v4fits[k[0:4]][(v4fits['r'].values>rthresh) ]

fig = scatter_w_marginals(threshFits['mori'], threshFits['sdori'],
                    'Mean Orientation', 'SD Orientation',
                    xlim= [0, 360],
                    #ylim= [0, np.round(np.max(threshFits['sdori'] ),-1)],
                    ylim = [0, 180],
                    title = 'r > ' + str(rthresh) + ', ' +str(len(threshFits)) + '/109' )

plt.savefig(top_dir + 'analysis/figures/images/v4_apc_ori.png')


fig = scatter_w_marginals(threshFits['mcurv'], threshFits['sdcurv'],
                    'Mean Curve', 'SD Curve',
                    xlim= [-1, 1],
                    ylim= [0, np.round(np.max(threshFits['sdcurv']),0)],
                    title = 'r > ' + str(rthresh))

plt.savefig(top_dir + 'analysis/figures/images/v4_apc_curv.png')

x = threshFits['cur_mean'].values
y = threshFits['cur_sd'].values
a = np.vstack((x,y)).T
b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
_, idx, counts = np.unique(b, return_index=True, return_counts=True)
unique_a = a[idx]
counts_s = (counts/np.double(max(counts))*100.)
plt.scatter(unique_a[:,0], unique_a[:,1], s=counts_s)


u_count_s = np.sort(np.unique(counts_s))
u_count = np.sort(np.unique(counts))

if len(u_count)>2:
    size = [np.min(u_count_s), np.median(u_count_s), np.max(u_count_s)]
    cnt = [np.min(u_count), int(np.median(u_count)), np.max(u_count)]
else:
    size = [np.min(u_count_s),  np.max(u_count_s)]
    cnt = [np.min(u_count), np.max(u_count)]

for_legend={'plt':[], 'label':[]}
_ = [for_legend['plt'].append(plt.scatter([],[], s=a_size)) for a_size in size]
_ = [for_legend['label'].append(str(a_cnt)) for a_cnt in cnt]


leg = plt.legend(for_legend['plt'], for_legend['label'], frameon=True,
                 fontsize='x-large', loc = 'upper left', handletextpad=1,
                 title='Counts', scatterpoints = 1)
'''