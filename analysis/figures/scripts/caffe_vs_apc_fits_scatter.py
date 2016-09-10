# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:36:34 2016

@author: dean
"""
import sys
import numpy as np

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import d_img_process as imp
import xarray as xr

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
    ax2.grid(True)
    x1 = min(plt.axis()[0:3:2])
    y1 = max(plt.axis()[1::2])
    plt.plot([x1, y1], [x1, y1], color='black')
    ax2.xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax2.yaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax3 = plt.subplot(gs[3])
    ax1.hist(y[~np.isnan(y)], orientation='horizontal', bins=bins)
    ax1.xaxis.set_major_locator(mtick.LinearLocator(numticks=2, presets=None))
    ax1.yaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax1.yaxis.set_label_text(titley)
    ax3.hist(x[~np.isnan(x)], orientation='vertical', bins=bins)
    ax3.yaxis.set_major_locator(mtick.LinearLocator(numticks=2, presets=None))
    ax3.xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax3.xaxis.set_label_text(titlex)
    ax3.set_xlim(xlim)
    ax1.set_ylim(ylim)
    #nice_axes(fig.axes)
    plt.show()
    return fig


def degen(daa):
    minfracvar = 0.5
    _ = (daa**2)
    tot_var = _.sum('shapes')
    non_zero = tot_var<1e-8
    just_one_shape = (_.max('shapes')/tot_var)>minfracvar
    degen_inds = just_one_shape + non_zero
    return degen_inds


figure_folder = top_dir + 'analysis/figures/images/'

da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
#daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
#daa=daa.loc[:, 0, :]#without translation
cnn_names =['APC362_deploy_fixing_relu_saved.prototxt_fixed_even_pix_width[24.0, 48.0]_pos_(64.0, 164.0, 51)bvlc_reference_caffenet' ]

daa = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].isel(scale=0).squeeze()
daa = daa.sel(x=daa.coords['x'][np.round(len(daa.coords['x'])/2.).astype(int)])
daa = daa.isel(shapes=da.coords['shapes']).chunk({})


da = da - da.mean('shapes')
da = da / da.vnorm('shapes')

#degen_inds = degen(daa)
#daa = daa[:,-degen_inds].chunk({})
daa = daa - daa.mean('shapes')
daa = daa / daa.vnorm('shapes')
daa = daa[:,daa.coords['layer_label'].values!='prob']
#shuffle(resp)
cov = np.dot(da.T, daa)
#plt.close('all')
#plt.subplot(211)
#plt.hist(np.max(cov,0), alpha=0.5, range=[0,1],bins=40,color='blue')
#plt.title('alex corr')
#plt.subplot(212)
#plt.hist(np.max(cov,1), alpha=0.5, range=[0,1],bins=40,color='red')
#plt.title('v4 corr')
plt.close('all')
plt.figure()
type_change = np.where(np.diff(daa.coords['layer'].values))[0]
type_label = daa.coords['layer_label'].values[type_change].astype(str)
bestFitInd = np.nanargmax((cov),1)
layer_num = daa.coords['layer'][bestFitInd]


layer_label = daa.coords['layer_label'].values
indexes = np.unique(layer_label, return_index=True)[1]
layer_label_ind = [layer_label[index] for index in sorted(indexes)]


bestFit = np.nanmax((cov),1)
plt.scatter(layer_num, bestFit)
plt.ylabel('Correlation')
plt.xticks(range(int(daa.coords['layer'].max().values+1)), layer_label_ind, rotation='vertical', size = 'small')
plt.ylim(0,1)
plt.xlim(-1,20)
plt.savefig(figure_folder + 'best_fit_toV4_and_layer_in_caffe.eps')


import pickle
with open(top_dir + 'data/models/ds_list_no_degen.p', 'rb') as f:
    ds_list= pickle.load(f)

#plt.figure()
#v4cor=np.nanmax(cov,1)
#v4apccor=ds_list[1]['real'].values
#plt.scatter(v4apccor, v4cor)
#plt.xlabel('APC fit to V4')
#plt.ylabel('AlexNet fit to V4')
#plt.gca().set_aspect('equal', adjustable='box')
#
#

scatter_w_marginals(v4apccor, v4cor, 'APC fit to V4', 'CaffeNet fit to V4', [0,1], [0,1], 'V4 fit to CaffeNet vs. APC', bins=40)
plt.savefig(figure_folder + 'apc_vs_caffe_net.eps')
