# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:23:36 2016

plots of v4 APC params

@author: dean
"""

import sys
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'v4cnn/common')
sys.path.append(top_dir + 'v4cnn/img_gen')
sys.path.append( top_dir + 'xarray/')

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
    cbar=plt.colorbar()
    cbar.ax.set_ylabel('Normalized Firing Rate', rotation= 270, labelpad=15, fontsize = 15,)
    cbar.ax.yaxis.set_ticks([0,.25,.5,.75, 1])
    cbar.ax.set_yticklabels(['0', .25, .5, .75, 1])
    cbar.solids.set_rasterized(True)
    plt.tight_layout()
    return data

#load images
baseImageDir = top_dir +'v4cnn/images/baseimgs/PCunique/'
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
cells = [108,]
v4 = xr.open_dataset(top_dir + 'v4cnn/data/responses/V4_362PC2001.nc')['resp']

#multiply shape luminance by firing rate
resp_sc = (v4[cells[0], : ].values*0.8 +.2)
imgStack = imgStack*resp_sc.reshape(362,1,1)

#sort images
sortStack = imgStack[list(reversed(np.argsort(resp_sc))),:,:]
sortStack = np.array([imp.centeredCrop(img, 64, 64) for img in sortStack])
data = vis_square(sortStack[0:image_square**2])
plt.title('Shape Response: Cell ' + str(cells[0]) )
plt.savefig(top_dir + 'v4cnn/analysis/figures/images/example_cell_rank_ordered_shapes.png')
plt.close('all')

v4fits = xr.open_dataset(top_dir + 'v4cnn/data/an_results/V4_370PC2001_LSQnonlin.nc')['fit']
v4fits = v4fits.to_pandas()
h = plt.hist(v4fits['r'].values)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Cell Count')
plt.xticks([0,0.25,0.5,0.75,1])
plt.xlim([0,1])
plt.ylim([0, 20])
plt.tight_layout()
plt.savefig(top_dir + 'v4cnn/analysis/figures/images/v4_apc_correlation_hist.png')
plt.close('all')

v4fits['mori'] = (v4fits['mori']/(2*np.pi))*360
v4fits['sdori'] = (v4fits['sdori']/(2*np.pi))*360
k = v4fits.keys()
rthresh = 0.6

threshFits = v4fits[k[0:4]][(v4fits['r'].values>rthresh) ]
plt.close('all')

fig = plt.figure()



import seaborn as sns

sns.set_context("talk", font_scale=1.4)
g = sns.jointplot(x='sdcurv', y='mcurv', data=threshFits, stat_func=None)
plt.title('Curvature Fits r>' + str(rthresh))
g.set_axis_labels('Standard Deviation','Mean')
g.ax_joint.set_ylim((-1,1.1))
g.ax_joint.set_xlim((0,1))
g.ax_joint.set_xticks([0,0.5,1])
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.gca().set_aspect('auto')
plt.savefig(top_dir + 'v4cnn/analysis/figures/images/v4_apc_fits.png')


#sns.set_context("talk", font_scale=1.4)
g = sns.jointplot(x='sdori', y='mori',data= threshFits, stat_func=None)
plt.title('Angular Position Fits r>' + str(rthresh))
g.set_axis_labels('Standard Deviation', 'Mean')
g.ax_joint.set_ylim((-8, 360))
g.ax_joint.set_xlim((0, 190))
g.ax_joint.set_xticks([0,90, 180])
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.gca().set_aspect('auto')

plt.savefig(top_dir + 'v4cnn/analysis/figures/images/v4_apc_correlation_hist.png')
