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
'''
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

maxAngSD = np.deg2rad(171); minAngSD = np.deg2rad(23)
maxCurSD = 0.98; minCurSD = 0.09;
maxCurSD = 0.98; minCurSD = 0.01
nMeans = 16; nSD = 16
fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dam = ac.make_apc_models(shape_dict_list, shape_id, fn, nMeans, nSD, maxAngSD, minAngSD,
                      maxCurSD, minCurSD, prov_commit=False, save=True, replace_prev_model=False)

#load the models you made, and fit them to the cells responses
dmod = xr.open_dataset(fn, chunks = {'models': 1000, 'shapes': 370}  )['resp']
cor = ac.cor_resp_to_model(da, dmod, fit_over_dims=None, prov_commit=False)

#correlation histogram for v4
plt.figure()
h = plt.hist(cor.values)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Cell Count')
plt.xticks([0,0.25,0.5,0.75,1])
plt.xlim([0,1])
plt.ylim([0, 20])
plt.tight_layout()
nice_axes(plt.gcf().axes)
plt.savefig(top_dir + 'analysis/figures/images/v4_apc_correlation_hist_brute.png')



#ALEXNET
if not 'daa' in locals():
    daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
    #ensure fraction of variance that can be explained by single translation
    #or shape is not more than minfracvar
    daa=daa.loc[:,0,:]
    minfracvar = 0.5
    _ = (daa**2)
    tot_var = _.sum('shapes')
    non_zero=tot_var<1e-8
    just_one_shape = (_.max('shapes')/tot_var)>minfracvar
    degen = just_one_shape + non_zero
    daa = daa[:,-degen]
    daa = daa - daa.mean(['shapes'])
    daa = daa.chunk({'unit':109, 'shapes':370}).T

#correlation histogram for v4
print('alex')


resp_n = daa.vnorm(('shapes'))
proj_resp_on_model = daa.dot(dmod)

resp_norm = resp_n
proj_resp_on_model_var = proj_resp_on_model
n_over = 1

all_cor = (proj_resp_on_model_var) / (resp_norm * (n_over**0.5))
corarg = all_cor.argmax('models')
model_fit_params = dmod.coords['models'][corarg]
cor = all_cor.max('models')
#cor = ac.cor_resp_to_model(daa, dmod, fit_over_dims=None, prov_commit=False)
#daa.dot(dmod).load()

plt.figure()
h = plt.hist(cor.values)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Unit Count')
plt.tight_layout()
nice_axes(plt.gcf().axes)
plt.savefig(top_dir + 'analysis/figures/images/alex_apc_correlation_hist_brute.png')


'''
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
plt.savefig(top_dir + 'analysis/figures/images/example_cell_rank_ordered_shapes.png')
'''