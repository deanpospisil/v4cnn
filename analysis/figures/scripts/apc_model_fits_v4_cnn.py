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

plt.close('all')
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
        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d_int))
        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))

def scatter_w_marginals(x, y, titlex, titley, xlim, ylim, title, bins=10):
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1] )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax2.scatter(x,y)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_title(title)
    ax2.xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax2.yaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax3 = plt.subplot(gs[3])
    ax1.hist(y, orientation='horizontal', bins=bins)
    ax1.xaxis.set_major_locator(mtick.LinearLocator(numticks=2, presets=None))
    ax1.yaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax1.yaxis.set_label_text(titley)
    ax3.hist(x, orientation='vertical', bins=bins)
    ax3.yaxis.set_major_locator(mtick.LinearLocator(numticks=2, presets=None))
    ax3.xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax3.xaxis.set_label_text(titlex)
    ax3.set_xlim(xlim)
    ax1.set_ylim(ylim)
    nice_axes(fig.axes)
    plt.show()
    return fig

#BRUTE FORCE FITS
#V4
import pickle
import apc_model_fit as ac

with open(top_dir + 'data/models/ds_list.p', 'rb') as f:
    ds_list= pickle.load(f)

da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
daa=daa.loc[:, 0, :]#without translation


plt.close('all')
fig = plt.figure()
nbins=20
n_resp_sets = len(ds_list)
for i, cor in enumerate(ds_list):
    ax = plt.subplot(n_resp_sets, 1, i+1)
    ax.hist(cor['real'].values, bins=nbins, range= [0,1])
    ax.hist(cor['shuf'].values, bins=nbins, range= [0,1])
    ax.xaxis.set_label_text('Correlation Coefficient')
    ax.yaxis.set_label_text('Unit Count', fontsize='x-large')
    #ax.tight_layout()
    ax.set_xlim([0,1])

nice_axes(plt.gcf().axes, nxticks=5, nyticks=5)
plt.tight_layout()
plt.show()

'''
#responses plotted on shapes V4
baseImageDir = top_dir +'images/baseimgs/PCunique/'
files = os.listdir(baseImageDir)
shapes = []
nImgs = 362
imgStack = np.zeros(( nImgs, 227, 227))

for f in files:
    if '.npy' in f:
        num, ext = f.split('.')
        num = int(num)
        imgStack[num, :,: ] = np.load(baseImageDir + f)
image_square = 10 #max 19
cells = [37,]
v4 = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc')['resp']
#multiply shape luminance by firing rate
resp_sc = (v4[cells[0], : ].values*0.8 +.2)
imgStack = imgStack*resp_sc.reshape(362,1,1)
#sort images
sortStack = imgStack[list(reversed(np.argsort(resp_sc))),:,:]
sortStack = np.array([imp.centeredCrop(img, 64, 64) for img in sortStack])
data = vis_square(sortStack[0:image_square**2])
plt.title('Shape Response: Cell ' + str(cells[0]) )
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/example_cell_rank_ordered_shapes.eps')


cor.coords['or_mean'] = ((cor.coords['or_mean']/(2*np.pi))*360)%360
cor.coords['or_sd'] = ((cor.coords['or_sd']/(2*np.pi))*360)

threshFits = cor[cor>rthresh].coords
ncell = len(threshFits['unit'])
fig = scatter_w_marginals(threshFits['or_mean'], threshFits['or_sd'],
                    'Mean Orientation', 'SD Orientation',
                    xlim= [0, 360],
                    #ylim= [0, np.round(np.max(threshFits['or_sd'] )+1,-1)],
                    ylim = [0, 180],
                    title = 'r > ' + str(rthresh) + ', ' +str(ncell) + '/109' )

plt.savefig(top_dir + 'analysis/figures/images/v4_apc_ori_brute.png')


fig = scatter_w_marginals(threshFits['cur_mean'].values, threshFits['cur_sd'].values,
                    'Mean Curve', 'SD Curve',
                    xlim= [-1, 1],
                    ylim= [0, np.round(np.max(threshFits['cur_sd']),0)],
                    title = 'r > ' + str(rthresh) + ', ' +str(ncell) + '/109' )

plt.savefig(top_dir + 'analysis/figures/images/v4_apc_curv_brute.png')

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
'''