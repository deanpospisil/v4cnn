# -*- coding: utf-8 -*-
"""
Created on Sun May 15 10:17:02 2016

@author: deanpospisil
"""

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
import numpy as np

from sklearn import cross_validation
from sklearn import datasets
from sklearn import linear_model




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

def tick_format_d(x,pos):
    if x==0:
        return('0')
    else:
        return(x)
def nice_axes(axes):
    for i, an_axes in enumerate(axes):
        an_axes.xaxis.set_tick_params(length=0)
        an_axes.yaxis.set_tick_params(length=0)
        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
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

#open those responses, and build apc models for their shapes
with open(top_dir + 'data/models/PC370_params.p', 'rb') as f:
    shape_dict_list = pickle.load(f)

da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']

shape_id = da.coords['shapes'].values
shape_dict_list = [shape_dict_list[sn] for sn in shape_id.astype(int)]

maxAngSD = np.deg2rad(171.); minAngSD = np.deg2rad(23.)
maxCurSD = 0.98; minCurSD = 0.09;
maxCurSD = 0.98; minCurSD = 0.01
nMeans = 10.; nSD = 10.
fn = top_dir + 'data/models/' + 'apc_models_362sm.nc'
dam = ac.make_apc_models(shape_dict_list, shape_id, fn, nMeans, nSD, maxAngSD, minAngSD,
                      maxCurSD, minCurSD, prov_commit=False, save=True, replace_prev_model=False)

#load the models you made, and fit them to the cells responses
dmod = xr.open_dataset(fn, chunks = {'models': 1000, 'shapes': 370}  )['resp']
#cor = ac.cor_resp_to_model(da, dmod, fit_over_dims=None, prov_commit=False)
X=dmod.values[:,:]
y=da.values.T[:,:]

#clf = linear_model.Lasso(alpha = 0.0005, positive=True)
#clf.fit( X, y)
#plt.plot(clf.coef_)


# Set a minimum threshold of 0.25
#model = linear_model.LassoLarsIC(criterion='aic').fit(X, y)
#m_log_alphas = model.alphas_
#plt.plot(m_log_alphas, model.criterion_, ':')
#plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
#         label='Average across the folds', linewidth=2)
a=np.argmin(np.array([np.linalg.lstsq(np.expand_dims(pred,1), y)[0] for pred in list(X[:,:100].T)]),0).T