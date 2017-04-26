# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:16:12 2017

@author: deanpospisil
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir + 'v4cnn'
import xarray as xr
import pandas as pd

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = cm.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in np.arange(N+1) ]
    # Return colormap object.
    return mpl.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def close_factors(n):
    factor_list = []
    for n_in in range(1,n):
        if (n%n_in) == 0:
            factor_list.append(n_in)
    factor_array = np.array(factor_list)
    paired_factors = np.array([factor_array, n/factor_array])
    paired_factors.shape
    best_ind = np.argmin(abs(paired_factors[1]-paired_factors[0]))
    closest_factors = paired_factors[:,best_ind]
    return closest_factors[0], closest_factors[1]

def net_vis_square(da, m=None, n=None):
    da = da.transpose('unit', 'y', 'x','chan')
    data = da.values
    if da.max()>1 or da.max()<0:
        print('Your image is outside the color range [0,1]')
        print('trying to fix it automatically')
        data = data - data.min([1, 2, 3], keepdims=True)
        data = data/data.max([1, 2, 3], keepdims=True)
    if type(m)==type(None):
        (m, n) = close_factors(da.shape[0])
        
    if data.shape[-1] == 1:
        data = np.repeat(data, 4, axis=-1)
    if data.shape[-1] == 3:
        data = np.concatenate([data, np.ones(np.shape(data)[:-1] + (1,))], axis=-1)
    
    if data.shape[1]<11:
        from scipy.ndimage import zoom
        data_new_size = np.zeros((np.shape(data)[0], 10, 10, 4))
        for i, im in enumerate(data):
            data_new_size[i, ...] = zoom(im, 
                         (2, 2, 1), 
                         order=0)
        data = data_new_size
            
    ypad = xpad = int(data.shape[1]/10)


    padding = ((0, 0), (ypad, ypad), (xpad, xpad), (0, 0))
    data = np.pad(data, padding, mode='constant', constant_values=0)
    data[...,-1] = 1
    #data = data.reshape(m*data.shape[1], n*data.shape[2], data.shape[3], order='C')
        # tile the filters into an image
    data = data.reshape((m, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    data = data.reshape((m * data.shape[1], n * data.shape[3], data.shape[4]))

    return data

def variance_to_power_ratio(da):
    red_dims = list(set(da.dims) - set(['unit',]))
    var = ((da-da.mean('chan'))**2).sum(red_dims)
    pwr =  (da**2).sum(red_dims)
    return var/pwr

def receptive_field(da):
    rf = (da**2).sum('chan')
    rf = rf.expand_dims('chan')
    rf = rf.transpose('unit', 'y', 'x','chan')
    return rf


def prin_comp_maps(da):
    da = da.transpose('unit', 'chan', 'y', 'x')

    data = da.values.reshape(da.shape[:2] + (np.product(da.shape[2:]),))
    u, s, v = np.linalg.svd(data, full_matrices=False)
    v = v.reshape(da.shape)
    
    u_da = xr.DataArray(u, dims=('unit', 'chan', 'pc'), 
                        coords=[range(n) for n in np.shape(u)])
    u_da.coords['unit'] = da.coords['unit']
    s_da = xr.DataArray(s, dims=('unit', 'sv'), 
                        coords=[range(n) for n in np.shape(s)])
    s_da.coords['unit'] = da.coords['unit']
    v_da = xr.DataArray(v, dims=('unit', 'pc', 'x', 'y'), 
                        coords=[range(n) for n in np.shape(v)])
    v_da.coords['unit'] = da.coords['unit']
    
    return u_da, s_da, v_da

def spatial_opponency(da):
    da = da.transpose('unit', 'chan', 'y', 'x')
    data = da.values.reshape(da.shape[:2] + (np.product(da.shape[2:]),))
    cov = np.matmul(data.transpose(0, 2, 1), data)
    cov = cov.sum(axis=(1,2)) - np.trace(cov, axis1=1, axis2=2)

    
    vnorm = np.linalg.norm(data, axis=1)
    outer_prod = (vnorm[:, :, np.newaxis])*(vnorm[:, np.newaxis, :])

    outer_prod = outer_prod.sum(axis=(1,2)) - np.trace(outer_prod, axis1=1, axis2=2)
    opponency = cov / outer_prod
    
    opponency_da = xr.DataArray(opponency, dims=('unit',))
    opponency_da.coords['unit'] = da.coords['unit']
    
    return opponency_da

def PC_spatial_freq(da):
    da = da.transpose('unit', 'chan', 'y', 'x')
    u_da, s_da, v_da = prin_comp_maps(da)
    
    fv = np.fft.fft2(v_da)
    a_fv = np.abs(fv)
    
    index = np.fft.fftfreq(a_fv.shape[-1])
    x, y = np.meshgrid(index, index)
    freq_power = (x**2 + y**2)**0.5
    freq_ori = np.angle(x + y*1j, deg=True)%180
    
    a_fv = np.fft.fftshift(a_fv, axes=(-2, -1))
    freq_power = np.fft.fftshift(freq_power)
    freq_ori = np.fft.fftshift(freq_ori)
    
    a_fv_da = xr.DataArray(a_fv, dims=('unit', 'pc', 'yft', 'xft'), 
                            coords=[range(n) for n in np.shape(a_fv)])
    unrav_a_fv = a_fv.reshape(a_fv.shape[:2] + (np.product(a_fv.shape[2:]),))
    
    #maybe change this to get spatial frequency up to a certain pc
    #what would relationship between spatial frequencies of different PC's mean?
    peak_freq_power= [freq_power.ravel()[ind[0]] for ind in np.argmax(unrav_a_fv,-1)]
    peak_freq_ori = [freq_ori.ravel()[ind[0]] for ind in np.argmax(unrav_a_fv,-1)]
    #peak_amp_frac = np.max(unrav_a_fv,-1)*2 / np.sum(unrav_a_fv, [-2, -1])
    
    
    keys = ['layer_label', 'unit']
    coord = [da.coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    spatial_freq = pd.DataFrame(np.vstack([peak_freq_power, peak_freq_ori,]).T, 
                                          index=index, columns=['amp', 'ori', ])
    return a_fv_da, spatial_freq

#%%
import pickle
goforit=False       
if 'netwts' not in locals() or goforit:
    with open(top_dir + '/nets/netwts.p', 'rb') as f:    
        try:
            netwts = pickle.load(f, encoding='latin1')
        except:
            netwts = pickle.load(f)
# reshape fc layer to be spatial
netwts[5][1] = netwts[5][1].reshape((4096, 256, 6, 6))
wts_by_layer = [layer[1] for layer in netwts]

net_resp_name = 'bvlc_reference_caffenety_test_APC362_pix_width[32.0]_x_(104.0, 124.0, 11)_x_(104.0, 124.0, 11)_amp_None.nc'
da = xr.open_dataset(top_dir + '/data/responses/' + net_resp_name)['resp']
if not type(da.coords['layer_label'].values[0]) == str:
    da.coords['layer_label'].values = [thing.decode('UTF-8') for thing in da.coords['layer_label'].values]
da.coords['unit'] = range(da.shape[-1])
#%%
from more_itertools import unique_everseen
layer_num = da.coords['layer']
layer_label_ind = da.coords['layer_label'].values
split_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',]
dims = ['unit','chan', 'y', 'x']
layer_names = list(unique_everseen(layer_label_ind))
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6',]


netwtsd = {}
for layer, name in zip(wts_by_layer, layer_names):
    dim_names = dims[:len(layer.shape)]
    layer_ind = da.coords['layer_label'].values == name 
    _ =  da[..., layer_ind].coords['unit']
    netwtsd[name] = xr.DataArray(layer, dims=dims, 
           coords=[range(n) for n in np.shape(layer)])
    netwtsd[name].coords['unit'] = _


conv1 = netwtsd['conv1']
conv1vis = conv1 - conv1.min(['chan', 'y', 'x'])
conv1vis = conv1vis/conv1vis.max(['chan', 'y', 'x'])
#conv1vis = conv1vis/conv1vis.max()

#conv1vis = conv1vis[:, :, :5, :5]
data = net_vis_square(conv1vis)
ax = plt.subplot(111)
ax.imshow(data, interpolation='nearest')
ax.set_xticks([])
ax.set_yticks([])
[ax.spines[pos].set_visible(False) for pos in ['left','right','bottom','top']]
plt.savefig(top_dir + '/analysis/figures/images/early_layer/1st_layer_filters.pdf')

#%%
da_ratio = variance_to_power_ratio(conv1)
u_da, s_da, v_da = prin_comp_maps(conv1)
rf = receptive_field(netwtsd['conv2'])
opponency_da = spatial_opponency(conv1)
a_fv_da, spatial_freq = PC_spatial_freq(conv1)

#%%
lw = 5
plt.style.use(top_dir+'/poster/dean_poster.mplstyle')
plt.figure()
da_ratio.plot.hist(cumulative=True, bins=100, histtype='step', lw=5)
da_ratio[:48].plot.hist(cumulative=True, bins=100, histtype='step',range=[0,1], lw=5)
da_ratio[48:].plot.hist(cumulative=True, bins=100, histtype='step', range=[0,1], lw=5)
plt.legend(['All Units', 'Group 1', 'Group 2'], loc=2)
plt.xlabel('Chromaticity');plt.ylabel('Filter\nCount', rotation=0, ha='right')
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/early_layer/hist.pdf', bboxinches='tight')

plt.figure()
ax, data = net_vis_square(conv1vis[da_ratio.argsort().values])
ax.set_ylabel('All Filters')
plt.figure()
ax, data = net_vis_square(conv1vis[:48][da_ratio[:48].argsort().values], m=4,n=12)
ax.set_ylabel('Group 1')

plt.figure()
ax, data = net_vis_square(conv1vis[48:][da_ratio[48:].argsort().values], m=4, n=12)
ax.set_ylabel('Group 2')

#%%%
#for variance explained maps for PC and variance explained make it a quantized color distribution
#favoring differentiation between higher correlation levels. Make the null hypothesis black.

#adapt sitmuli generation for two shapes, show that the TI ones also are the ones that increase
#their response to a shape at any position.

rfperc = rf 
rfperc = rfperc/rfperc.sum(['x', 'y'])
N = 5
c_disc = cmap_discretize(mpl.cm.plasma, N=N)

vmax = 0.2
vmin = 0 
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
rfvis_dat = np.squeeze(mpl.cm.ScalarMappable(cmap=c_disc, norm=norm).to_rgba(rfperc))
rf_vis = xr.DataArray(rfvis_dat, dims=rf.dims)

ax, data = net_vis_square(rf_vis)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0, 0.7, 1, 1])
plt.imshow(data)
[ax.spines[pos].set_visible(False) for pos in ['left','right','bottom','top']]
ax.set_xticks([]);ax.set_yticks([]);

ax = fig.add_axes([0.05, 0.63, 0.9, 0.05])
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=c_disc,
                                norm=norm,
                                orientation='horizontal',
                                extend='max',
                                ticks=np.linspace(vmin, vmax, N+1))
cb1.set_label('Percent Variance')

#%%
plt.style.use('seaborn-paper')
conv2 = netwtsd['conv2']
freq = 2
lay1_1 = np.deg2rad(spatial_freq['ori'][:48])
lay2_1 = conv2.values.reshape(conv2.shape[:2] + (np.product(conv2.shape[2:]),))
lay2_1 = lay2_1[:128]
#our predictors are a sinusoidal function the preferred orientation of the prior layer
A = np.vstack([np.cos(lay1_1*freq), np.sin(lay1_1*freq), np.ones(len(lay1_1))]).T

fit = [np.linalg.lstsq(A, a_filt)[:2] for a_filt in lay2_1]
reg_coefs = np.array([a_fit[0] for a_fit in fit])
residual = np.array([a_fit[1] for a_fit in fit])

res_map = residual.reshape((lay2_1.shape[0],) + conv2.shape[-2:])

#fit_maps = fits.reshape((lay2_1.shape[0],) + (A.shape[1],) + conv2.shape[-2:])

prediction = np.matmul(A, reg_coefs)

lay2_1_nrm = lay2_1 - lay2_1.mean(-2, keepdims=True)
lay2_1_nrm = lay2_1_nrm/(np.sum(lay2_1_nrm**2, -2, keepdims=True)**0.5)

prediction_nrm = prediction - prediction.mean(-2, keepdims=True)
prediction_nrm = prediction_nrm/(np.sum(prediction_nrm**2, -2, keepdims=True)**0.5)


corr_map = np.sum(lay2_1_nrm*prediction_nrm, -2)
best_ind = np.unravel_index(np.argmax(corr_map), corr_map.shape)
p = prediction[best_ind[0], :, best_ind[1]]
b = lay2_1[best_ind[0], :, best_ind[1]]
#plt.scatter(np.sort(spatial_freq['ori'][:48]), b[sort_ori])
plt.hist(corr_map.ravel())
#%%



#%%
sort_ori = np.argsort(spatial_freq['ori'][:48]).values
plt.scatter(np.sort(spatial_freq['ori'][:48]),lay2_1[best, sort_ori, best_spot], s=10)
    
prediction[best, sort_ori, best_spot]

    