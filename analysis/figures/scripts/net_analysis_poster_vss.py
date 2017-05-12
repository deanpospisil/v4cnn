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
plt.style.use(top_dir + '/poster/dean_poster.mplstyle')
from math import log10, floor
plt.rc('text', usetex=False)


def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))
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
        cmap = mpl.cm.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red', 'green', 'blue')):
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

def net_vis_square_da(da, m=None, n=None):
    da = da.transpose('unit', 'y', 'x','chan')
    data = da.values
    if da.max()>1 or da.max()<0:
        print('Your image is outside the color range [0,1]')
        print('trying to fix it automatically')
        data = data - data.min((1, 2, 3), keepdims=True)
        data = data/data.max((1, 2, 3), keepdims=True)
    if type(m)==type(None):
        (m, n) = close_factors(da.shape[0])
        m = int(m)
        n = int(n)
        
    if data.shape[-1] == 1:
        data = np.repeat(data, 4, axis=-1)
    if data.shape[-1] == 3:
        data = np.concatenate([data, np.ones(np.shape(data)[:-1] + (1,))], axis=-1)
    
    if data.shape[1]<11:
        from scipy.ndimage import zoom
        zoom_amt = np.ceil(9/data.shape[1])
        print(zoom_amt)
        data_new_size = np.zeros((np.shape(data)[0], int(zoom_amt*data.shape[1]), 
                                      int(zoom_amt*data.shape[1]), 4))
        for i, im in enumerate(data):
            data_new_size[i, ...] = zoom(im, 
                         (zoom_amt, zoom_amt, 1), 
                         order=0)
        data = data_new_size
            
    ypad = xpad = int(data.shape[1]/5)


    padding = ((0, 0), (ypad, ypad), (xpad, xpad), (0, 0))
    data = np.pad(data, padding, mode='constant', constant_values=0)
    data[...,-1] = 1
    #pad_data = data
    #data = data.reshape(m*data.shape[1], n*data.shape[2], data.shape[3], order='C')
        # tile the filters into an image
    data = data.reshape((m, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    data = data.reshape((m * data.shape[1], n * data.shape[3], data.shape[4]))
    return data
#%%
def clean_imshow(da, ax=None):
    if ax == None:
        ax = plt.subplot(111)
    ax.imshow(data, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[pos].set_visible(False) for pos in ['left','right','bottom','top']]
    return ax

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
    v = v.reshape(v.shape[:2]+ da.shape[-2:])#reshape into space
    
    u_da = xr.DataArray(u, dims=('unit', 'chan', 'pc'), 
                        coords=[range(n) for n in np.shape(u)])
    u_da.coords['unit'] = da.coords['unit']
    s_da = xr.DataArray(s, dims=('unit', 'sv'), 
                        coords=[range(n) for n in np.shape(s)])
    s_da.coords['unit'] = da.coords['unit']
    v_da = xr.DataArray(v, dims=('unit', 'chan', 'x', 'y'), 
                        coords=[range(n) for n in np.shape(v)])
    v_da.coords['unit'] = da.coords['unit']
    
    return u_da, s_da, v_da

def prin_comp_rec(da, n_pc=2):
    da = da.transpose('unit', 'chan', 'y', 'x')
    u_da, s_da, v_da = prin_comp_maps(da)
    u, s, v = (u_da.values, s_da.values, v_da.values)
    v = v.reshape(v.shape[:2] + (np.product(da.shape[-2:]),))#unrwap
    S = np.array(list(map(np.diag, s[:, :n_pc,])))
    coefs = np.matmul(S, v[:, :n_pc, :])
    reconstruction = np.matmul(u[..., :n_pc], coefs)
    
    reconstruction = reconstruction.reshape(reconstruction.shape[:2] + da.shape[-2:])
    reconstruction_da = xr.DataArray(reconstruction, dims=da.dims, 
                        coords=[range(n) for n in np.shape(da)])
    reconstruction_da.coords['unit'] = da.coords['unit']
    
    coefs = coefs.reshape(coefs.shape[:2] + da.shape[-2:])
    coefs_da = xr.DataArray(coefs, dims=da.dims, 
                        coords=[range(n) for n in np.shape(coefs)])
    coefs_da.coords['unit'] = da.coords['unit']
    
    return coefs_da, reconstruction_da
    

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

def polar2cart(r, theta, center):

    x = r  * np.cos(theta) + center[0]
    y = r  * np.sin(theta) + center[1]
    return x, y

def img2polar(img, center, final_radius, initial_radius = None, phase_width = 3000):

    if initial_radius is None:
        initial_radius = 0

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), 
                            np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    if img.ndim ==3:
        polar_img = img[Ycart,Xcart,:]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))

    return polar_img

def PC_spatial_freq(da, nomean=True):
    da = da.transpose('unit', 'chan', 'y', 'x')
    u_da, s_da, v_da = prin_comp_maps(da)
    v_da_0 = v_da[:, 0]
    
    if nomean:
        v_da_0 = v_da_0 - v_da_0.mean(['x','y'])
    
    min_upsamp_size =  100 
    upsamp_rate = np.ceil(min_upsamp_size/np.min(np.shape(v_da)[1:]))
    (upsamp_y, upsamp_x) = np.round(np.array(np.shape(v_da_0)[1:])*upsamp_rate).astype(int)
    
    fv_us = np.fft.fft2(v_da_0, s=(upsamp_y, upsamp_x))
    a_fv_us = np.abs(fv_us)
    a_fv_us = np.fft.fftshift(a_fv_us, axes=(-2, -1))
    
    fv = np.fft.fft2(v_da_0)
    a_fv = np.abs(fv)
    a_fv = np.fft.fftshift(a_fv, axes=(-2, -1))


    
    radius = np.max([np.round(upsamp_y/2), np.round(upsamp_y/2)]).astype(int)
    polar_a_fv = np.array([img2polar(filt, [np.round(upsamp_y/2).astype(int), 
                                            np.round(upsamp_x/2).astype(int)],
                                            radius,
                                            phase_width=360)[:, :180] 
                                            for filt in a_fv_us])
    prfrd_ori = np.deg2rad(polar_a_fv.sum(1).argmax(1))
    prfrd_amp = polar_a_fv.sum(2).argmax(1)/np.double(radius)
    
    keys = ['layer_label', 'unit']
    coord = [da.coords[key].values for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)
    spatial_freq = pd.DataFrame(np.vstack([prfrd_amp, prfrd_ori,]).T, 
                                          index=index, columns=['amp', 'ori', ])

    a_fv_da = xr.DataArray(a_fv, dims=('unit', 'y', 'x'), 
                            coords=[range(n) for n in np.shape(a_fv)])
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
#conv1 = window.reshape(1,1,11,11)*conv1
conv1vis = conv1 - conv1.min(['chan', 'y', 'x'])
conv1vis = conv1vis/conv1vis.max(['chan', 'y', 'x'])

#conv1vis = conv1vis/conv1vis.max()

#conv1vis = conv1vis[:, :, :5, :5]
data = net_vis_square_da(conv1vis)
ax = clean_imshow(data)

plt.savefig(top_dir + '/analysis/figures/images/early_layer/1st_layer_filters.pdf', bbox_inches='tight')

#%%
u_da, s_da, v_da = prin_comp_maps(netwtsd['conv2'])
da_ratio = variance_to_power_ratio(conv1)
u_da, s_da, v_da = prin_comp_maps(conv1)
rf = receptive_field(netwtsd['conv2'])
opponency_da = spatial_opponency(conv1)

#%%
c = da_ratio.values
wts_c = conv1.transpose('unit', 'y','x', 'chan').values

#our reference points for plotting in the 2-d space
rgb = np.eye(3) 
axis1 = np.real(np.fft.ifft([0,1,0]))
axis2 = np.imag(np.fft.fft([0,1,0]))
proj_mat = np.vstack([axis1 ,axis2]).T
proj_mat /= np.sum(proj_mat**2,0, keepdims=True)**0.5
rgb_proj = np.dot(rgb, proj_mat)/1.
fig, axs = plt.subplots(figsize=(12,4), nrows=4, ncols=12)
#for ax, c_ind in zip(axs.ravel(), c.argsort()[::-1]):
for  c_ind, ax in zip(list(range(96))[48:], axs.ravel()):
    to_proj = wts_c[c_ind].squeeze()
    amp = np.max(np.abs(to_proj))
    to_proj = to_proj/amp
    to_proj = to_proj.reshape(np.product(to_proj.shape[:-1]), 3)
    proj = np.dot(to_proj, proj_mat)
    color = np.array(to_proj - np.min(to_proj))
    color /= np.max(color)
    _=ax.scatter(proj[:, 0], proj[:, 1], c=color, edgecolors='None', s=12)
    _=plt.axis('equal')
    #ax.set_title('i='+str(c_ind) +' \nc='+ str(np.round(c[c_ind],2)) +
    #             '\n amp=' + str(np.round(amp,2)), fontsize=6)
    for spine in ['left', 'right', 'bottom', 'top']:
        ax.spines[spine].set_visible(True)
    _ = ax.scatter(rgb_proj[:,0], rgb_proj[:,1], c='None', edgecolors=rgb)
    _ = ax.scatter(0, 0, c='None', edgecolors='k')
    ax.text(x=0.95, y=0.95, s=str(np.round(opponency_da[c_ind].values,2)), 
            ha='right', va='top', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.5, pad=2), fontsize=12)
for ax in axs.ravel():
    ax.set_xticks([]);ax.set_yticks([]);ax.axis('equal')
plt.tight_layout(w_pad=0.01, h_pad=0.01)
plt.savefig(top_dir + '/analysis/figures/images/early_layer/1st_layer_filters_chroma.pdf')

#%%

a_fv_da, spatial_freq = PC_spatial_freq(conv1, nomean=False)
a_fv_da_nrm = a_fv_da/a_fv_da.max(['x', 'y'])
spec_dat = np.squeeze(mpl.cm.ScalarMappable(cmap=mpl.cm.plasma).to_rgba(np.expand_dims(a_fv_da_nrm.values, -1)))
spec_vis = xr.DataArray(spec_dat, dims=('unit','y', 'x', 'chan'))

data = net_vis_square_da(spec_vis)
fig = plt.figure()
clean_imshow(data)
#ax = fig.add_axes([0.15, 0.01, 0.7, 0.05])
#cb1 = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.plasma,
 #                               orientation='horizontal',
#                                ticks=np.linspace(0, 1, 6))
#cb1.set_label('Fraction Amplitude')
plt.savefig(top_dir + '/analysis/figures/images/early_layer/spec_conv1.pdf', bbox_inches='tight')

#%%%
plt.figure(figsize=(5,4))
not_dc = spatial_freq['amp']>0
plt.scatter(np.rad2deg(spatial_freq['ori'][not_dc]), spatial_freq['amp'][not_dc], alpha=0.5, edgecolor='k',color='none')
plt.xticks([0,90,180])
plt.yticks([0,0.5,1])
plt.xlabel('Orientation')
plt.ylabel('Spatial Frequency')
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/early_layer/ori_freq_plot.pdf')

#%%
lw = 4
plt.style.use(top_dir + '/poster/dean_poster.mplstyle')

plt.figure(figsize=(6,5))
da_ratio.plot.hist(cumulative=True, bins=100, histtype='step', lw=lw)
da_ratio[:48].plot.hist(cumulative=True, bins=100, histtype='step',range=[0,1], lw=lw)
da_ratio[48:].plot.hist(cumulative=True, bins=100, histtype='step', range=[0,1], lw=lw)
plt.legend(['All Units', 'Group 1', 'Group 2'], loc=2)
plt.yticks([0,24,48, 72, 96])
plt.xlabel('Chromaticity');plt.ylabel('Filter\nCount', rotation=0, ha='right')
plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/early_layer/chrom_hist.pdf', bbox_inches='tight')
#%%
plt.figure()
data = net_vis_square_da(conv1vis[da_ratio.argsort().values])
ax = clean_imshow(da)
ax.set_ylabel('All Filters')
plt.figure()
data = net_vis_square_da(conv1vis[:48][da_ratio[:48].argsort().values], m=4,n=12)
ax = clean_imshow(da)
ax.set_ylabel('Group 1')
plt.savefig(top_dir + '/analysis/figures/images/early_layer/group1.pdf', bbox_inches='tight')


plt.figure()
data = net_vis_square_da(conv1vis[48:][da_ratio[48:].argsort().values], m=4, n=12)
ax = clean_imshow(da)
ax.set_ylabel('Group 2')
plt.savefig(top_dir + '/analysis/figures/images/early_layer/group2.pdf', bbox_inches='tight')

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

data = net_vis_square_da(rf_vis)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.2, 0.7, 0.7])
clean_imshow(data ,ax)
ax = fig.add_axes([0.25, 0.15, 0.4, 0.03])
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=c_disc,
                                norm=norm,
                                orientation='horizontal',
                                extend='max',
                                ticks=np.linspace(vmin, vmax, N+1))
cb1.set_label('Percent Variance')
plt.savefig(top_dir + '/analysis/figures/images/early_layer/rfconv2.pdf', bbox_inches='tight')


#%%
plt.style.use(top_dir + '/poster/dean_poster.mplstyle')

rf_list = [receptive_field(netwtsd[layer]) for layer in layer_names]
rf_list = [layer/layer.sum(['x', 'y']) for layer in rf_list]

m = len(rf_list)
n = 1

plt.figure(figsize=(2,8))

for layer, n in zip(rf_list,  range(m)):
    ax = plt.subplot(m, 1, n+1)
    lower_bound = 1./layer.shape[1]**2.

    #ax.annotate(str(layer.layer_label[0].values), [0.7,0.7], xycoords='axes fraction', fontsize=16)
    ax.set_xticks([0, lower_bound, 0.25, 0.5])
    ax.set_xticklabels([])
    if n==0:
        ''
        #ax.set_ylabel('Count', labelpad=4) 
    if n==m-1:
        ax.set_xticklabels(['','l.b.', '0.25', '0.5'])
        ax.set_xlabel('Fraction RF\nVariance of Max')
        
    _ = layer.groupby('unit').max().values
    ax.hist(_, normed=0, bins=50, range=[0,.5], align='right')
    ax.set_xlim(0, 0.5)


plt.savefig(top_dir + '/analysis/figures/images/early_layer/all_layer_rf.pdf', bbox_inches='tight')

 #%%
def cor_over(da1, da2, center_dims, cor_dims):
    das = [da1, da2]
    das = [da - da.mean(center_dims) for da in das]
    das = [da / (da**2).sum(cor_dims)**0.5 for da in das]
    da_cor = (das[0]*das[1]).sum(cor_dims)
    return da_cor


def reg_on_chan_weights(da, A):
    da = da.transpose('unit', 'chan', 'y', 'x')
    lay2 = da.values.reshape(da.shape[:2] + (np.product(da.shape[2:]),)) 
    fit = [np.linalg.lstsq(A, a_filt)[:2] for a_filt in lay2]
    reg_coefs = np.array([a_fit[0] for a_fit in fit])

    prediction = np.matmul(A, reg_coefs).reshape(da.shape)
    prediction = xr.DataArray(prediction, dims=da.dims, coords=da.coords)
    
    da_sum_cor = cor_over(da, prediction, ['chan'], ['chan', 'x','y'])
    da_cor_map = cor_over(da, prediction, ['chan'], ['chan']).expand_dims('chan',0)
    
    return da_cor_map, da_sum_cor, reg_coefs
conv2 = netwtsd['conv2']
freq = 2
lay1_1 = spatial_freq['ori'][:48]
sort_ori = np.argsort(lay1_1)
lay2_1 = conv2.values.reshape(conv2.shape[:2] + (np.product(conv2.shape[2:]),))
lay2_1 = lay2_1[:128]
#our predictors are a sinusoidal function the preferred orientation of the prior layer
A = np.vstack([np.cos(lay1_1*freq), np.sin(lay1_1*freq), np.ones(len(lay1_1))]).T

da_cor_map, da_sum_cor, reg_coefs = reg_on_chan_weights(conv2[:128], A[:48])  

rads = np.linspace(0, np.pi, 100)
A_smth = np.vstack([np.cos(rads*freq), np.sin(rads*freq), np.ones(len(rads))]).T
smooth_prediction = np.matmul(A_smth, reg_coefs)

cor_map_dat = da_cor_map.squeeze().values.reshape((da_cor_map.shape[1],)
                                    + (np.product(da_cor_map.shape[2:]),))
cor_level_loc = []
cor_level = []
cor_level_near = [0.3,0.4,0.5,0.6,0.7,0.85][::-1]
for level in cor_level_near:
   the_loc = np.unravel_index(np.argmin(np.abs(cor_map_dat-level)), cor_map_dat.shape)
   cor_level_loc.append(the_loc)
   cor_level.append(cor_map_dat[the_loc[0], the_loc[1]])

import matplotlib.gridspec as gridspec
m = len(cor_level_near)
n = 1

# We'll use two separate gridspecs to have different margins, hspace, etc
gs_top = plt.GridSpec(m, 1, top=1, left=0.4)
gs_base = plt.GridSpec(m, 1, hspace=0.7, left=0.4)
fig = plt.figure(figsize=(4,12))

# Top (unshared) axes
topax = fig.add_subplot(gs_top[0,:])
# The four shared axes
ax = fig.add_subplot(gs_base[1,:]) # Need to create the first one to share...
other_axes = [fig.add_subplot(gs_base[i,:], sharex=ax) for i in range(2, m)]
bottom_axes = [ax] + other_axes


for ind, cor, n in zip(cor_level_loc, cor_level,  range(m)):
    if n==0:
        ax = topax
        ax.set_title('R = ' +np.str(np.round(cor, 2)))
        ax.set_ylabel('Conv2\nWeight', rotation=0, labelpad=1, va='center', ha='right')
        ax.set_xlabel('Orientation Conv1')
        ax.set_xticks([0,90,180])
        
    else:
        ax = bottom_axes[n-1]
        ax.set_title(np.str(np.round(cor, 2)))
        ax.set_xticks([0,90,180])
        ax.set_xticklabels([])

    ax.grid(b=True)
    b = lay2_1[ind[0], :, ind[1]]
    ax.scatter(np.rad2deg(spatial_freq['ori'][:48][sort_ori]), b[sort_ori], s=4)
    max_extent = round_to_1(np.max([np.max(b), np.abs(np.min(b))]))
    ytick = [-max_extent, 0, max_extent]
    ax.set_yticks(ytick)
    ytick[1] = '0'
    ax.set_yticklabels(ytick)
    ax.set_ylim([-max_extent*1.2, max_extent*1.2])
    ax.plot(np.rad2deg(rads), smooth_prediction[ind[0],:,ind[1]], color='b', lw=1)

    #plt.hist(corr_map.ravel(), histtype='step', range=[0,1])
plt.savefig(top_dir + '/analysis/figures/images/early_layer/cross_examples.pdf', bbox_inches='tight')

##%%
#space_values = np.linspace(-1,1,11)
#x, y = np.meshgrid(space_values, space_values)
#dist = (x**2+y**2)**0.5
#sigma = 0.5
#window = (1./(2*np.pi*sigma**2)**0.5)*np.exp(-(dist**2)/(2*sigma**2))
#a_fv_da, spatial_freq = PC_spatial_freq(window.reshape(1,1,11,11)*conv1)

#%%
a_fv_da, spatial_freq = PC_spatial_freq(conv1)
da_cor_map_lst = []

#null_cor_map_lst = []
#shuffle_ind = range(len(spatial_freq))

freqs = np.linspace(0.1, 12, 100)
print(freqs)
lay1_1 = spatial_freq['ori']

for freq in freqs:
    #our predictors are a sinusoidal function the preferred orientation of the prior layer
    A = np.vstack([np.cos(lay1_1*freq), np.sin(lay1_1*freq), np.ones(len(lay1_1))]).T
    #A = np.vstack([np.cos(lay1_1*freq), np.sin(lay1_1*freq),]).T

    da_cor_map1, da_sum_cor1, reg_coefs = reg_on_chan_weights(conv2[:128], A[:48])  
    da_cor_map2, da_sum_cor2, reg_coefs = reg_on_chan_weights(conv2[128:], A[48:]) 
    da_cor_map = xr.concat([da_sum_cor1, da_sum_cor2], dim='unit')
    da_cor_map_lst.append(da_cor_map)
#%%
plt.style.use('default')
plt.style.use(top_dir + '/poster/dean_poster.mplstyle')
plt.figure(figsize=(4,4))
da_cor_map_freq = xr.concat(da_cor_map_lst, dim='freq').squeeze()
da_cor_map_freq['freq'] = freqs

plt.subplot(111)
for perc in [0.75, 0.5, 0.25]:
    da_cor_map_freq[:, :128].quantile(perc, [ 'unit']).plot()
for i, freq in list(enumerate(freqs))[::4]:
    plt.scatter([freq,]*len(da_cor_map_freq[i, :128]), 
                da_cor_map_freq[i, :128], s=2, edgecolors='None', 
                c='k', alpha=0.2)
plt.title('Sinusoidal Fit to Weights\nConv2 Group 1')
plt.xticks(np.linspace(0,12,7))
plt.legend(['75th', '50th', '25th'], title='Percentile', loc=1, borderaxespad=0)
plt.ylim(0,1);
plt.yticks([0,0.5,1])
plt.xlabel('Frequency (cycles/radian)')
plt.ylabel('Correlation')

#plt.subplot(212)
#for perc in [0.75, 0.5, 0.25]:
#    da_cor_map_freq[:, 128:].quantile(perc, [ 'unit']).plot()
#for i, freq in list(enumerate(freqs))[::4]:
#    plt.scatter([freq,]*len(da_cor_map_freq[i, :128]), 
#                da_cor_map_freq[i, 128:], s=2, edgecolors='None', 
#                c='k', alpha=0.2)
#plt.title('Conv2 Group 2') 
#plt.xticks(np.linspace(0,12,7))
#plt.yticks([0,0.5,1])
#plt.gca().set_yticklabels([])
#plt.gca().set_xticklabels([])

#plt.ylim(0,1)
#plt.xlabel('')
plt.tight_layout()


plt.savefig(top_dir + '/analysis/figures/images/early_layer/cor_cross_ori_spec.pdf', bbox_inches='tight')

#%%  
plt.figure()
conv2 = netwtsd['conv2']
freq = 2
lay1_1 = np.deg2rad(spatial_freq['ori'])
#our predictors are a sinusoidal function the preferred orientation of the prior layer
A = np.vstack([np.cos(lay1_1*freq), np.sin(lay1_1*freq), np.ones(len(lay1_1))]).T

da_cor_map1, da_sum_cor, reg_coefs = reg_on_chan_weights(conv2[:128], A[:48])  
da_cor_map2, da_sum_cor, reg_coefs = reg_on_chan_weights(conv2[128:], A[48:])  

N = 7
c_disc = cmap_discretize(mpl.cm.plasma, N=N)
vmax = 1
vmin = 0.3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

cormap_dat1 = np.squeeze(mpl.cm.ScalarMappable(cmap=c_disc, norm=norm).to_rgba(da_cor_map1))
cormap_dat2 = np.squeeze(mpl.cm.ScalarMappable(cmap=c_disc, norm=norm).to_rgba(da_cor_map2))

cormap_vis1 = xr.DataArray(cormap_dat1, dims=('unit', 'y', 'x', 'chan'))
cormap_vis2 = xr.DataArray(cormap_dat2, dims=('unit', 'y', 'x', 'chan'))

data1 = net_vis_square_da(cormap_vis1)
data2 = net_vis_square_da(cormap_vis2)

data = np.vstack([data1,data2])

fig = plt.figure()
ax = fig.add_axes([0.1, 0.2, 0.7, 0.7])
clean_imshow(data ,ax)
plt.title('Cross-Orientation Suppresion Maps')

ax = fig.add_axes([0.25, 0.15, 0.4, 0.03])
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=c_disc,
                                norm=norm,
                                orientation='horizontal',
                                extend='min',
                                ticks=np.linspace(vmin, vmax, N+1))
cb1.set_label('Correlation')
plt.savefig(top_dir + '/analysis/figures/images/early_layer/cor_cross_ori.pdf', bbox_inches='tight')
#%%

plt.figure(figsize=(3,3))
conv2 = netwtsd['conv2']
u_da, s_da, v_da = prin_comp_maps(netwtsd['conv2'])

frac_var = ((s_da.isel(sv=[0,1])**2).sum('sv')/(s_da**2).sum('sv'))
(frac_var).plot.hist(histtype='bar', 
lw=3, range=[0,1], bins=20, cumulative=False, normed=False)
plt.title('Two PC\nReconstruction')
plt.xlabel('Fraction Variance\nExplained')
plt.ylabel('Number\nConv2 Units')
#plt.grid()
#plt.yticks([0, 0.5, 1])
#plt.gca().set_yticklabels(['0',  '0.5', '1'])
plt.xticks([0, 0.25, 0.5, 0.75, 1])
plt.gca().set_xticklabels(['0', '','0.5', '', '1'])
plt.xlim(0,1)
plt.savefig(top_dir + '/analysis/figures/images/early_layer/two_pc_recon.pdf', bbox_inches='tight')
#%%
plt.style.use(top_dir + '/poster/dean_poster.mplstyle')



s_list = [prin_comp_maps(netwtsd[layer])[1] for layer in layer_names]
frac_var_list = [((s.isel(sv=[0])**2).sum('sv')/(s**2).sum('sv')) for s in s_list]

m = len(frac_var_list)
n = 1

plt.figure(figsize=(2,8))

for layer, n in zip(frac_var_list,  range(m)):
    ax = plt.subplot(m, 1, n+1)

    if n==0:
        #ax.set_ylabel('Count', labelpad=4) 
        ax.set_xticks([0, 0.5 ,1])
        ax.set_xticklabels([])
    else:
        ax.set_xticks([0, 0.5 ,1])
        ax.set_xticklabels([])
        
    if n==m-1:
        ax.set_xticks([0,0.5,1])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_xlabel(r'$\frac{\lambda_1^2}{\sum{\lambda_i^2}}$')
        
    ax.hist(layer.values, lw=3, range=[0,1], bins=50, cumulative=False, normed=0,)
    ax.set_xlim(0,1)
    ax.annotate(str(layer.layer_label[0].values), [1.05,0.5], 
                xycoords='axes fraction', fontsize=16)
plt.savefig(top_dir + '/analysis/figures/images/early_layer/deep_layers_pc1.pdf',
            bbox_inches='tight')

#%%
import husl

def cart2angle(a):
    ang = np.array([np.arctan2(cart[0, :], cart[1, :]) for cart in a ])
    ang = np.rad2deg(ang)%360
    return ang
def cart2mag(a):
    mag = np.sum(a**2, 1)**0.5
    return mag
def cart2pol(a):
    angle = cart2angle(a)
    mag = cart2mag(a)
    pol = np.dstack((angle, mag)).swapaxes(1,2)
    return pol
def ziphusl(a):
    rgb = husl.huslp_to_rgb(a[0], a[1], a[2])
    return rgb 
plt.figure()
sat_scale = 100
cor_scale = 80
conv2 = conv2.transpose('unit', 'chan', 'y', 'x')
coefs_da, reconstruction_da = prin_comp_rec(conv2, n_pc=2)

da_cor = cor_over(conv2, reconstruction_da, ['chan'], ['chan'])
da_cor = da_cor.expand_dims('chan')

N = 7
c_disc = cmap_discretize(mpl.cm.plasma, N=N)
vmax = 1
vmin = 0.3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

cormap_dat1 = np.squeeze(mpl.cm.ScalarMappable(cmap=c_disc, norm=norm).to_rgba(da_cor))

cormap_vis1 = xr.DataArray(cormap_dat1, dims=('unit', 'y', 'x', 'chan'))

data = net_vis_square_da(cormap_vis1)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.2, 0.7, 0.7])
clean_imshow(data ,ax)
plt.title('1st PC corr')
ax = fig.add_axes([0.25, 0.15, 0.4, 0.03])
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=c_disc,
                                norm=norm,
                                orientation='horizontal',
                                extend='min',
                                ticks=np.linspace(vmin, vmax, N+1))
cb1.set_label('Correlation')
plt.savefig(top_dir + '/analysis/figures/images/early_layer/pc_correlation.pdf', bbox_inches='tight')

#%%
sat_scale = 100
cor_scale = 83
coefs_da_c = coefs_da[:, 0, ...]*1j + coefs_da[:, 1, ...]
angle = xr.ufuncs.angle(coefs_da_c, deg=True)
#angle = (angle + 2 * np.pi) % (2 * np.pi)
mag = np.abs(coefs_da_c)*200
mag[...] = sat_scale
lum = da_cor.squeeze()*cor_scale


husl_coefs = xr.concat([angle, mag, lum], dim='husl')

coeffs_pol_rgb = np.apply_along_axis(ziphusl, 0, husl_coefs)
coeffs_pol_rgb = xr.DataArray(coeffs_pol_rgb, dims=['chan', 'unit', 'y', 'x'])
data = net_vis_square_da(coeffs_pol_rgb)
clean_imshow(data)
plt.title('Hue=Angle(PC1 Coef., PC2 Coef.)\nLuminance=Correlation(Reconstruction, Original)')

plt.savefig(top_dir + '/analysis/figures/images/early_layer/layer2_pc_vis.pdf',bbox_inches='tight')

#%%
plt.rc('text', usetex=False)
opponency_da = spatial_opponency(conv2)
da_sum_cor = cor_over(conv2, reconstruction_da, ['chan'], ['chan', 'x','y'])
use_cor = True
plt.figure(figsize=(8,8))

ax1 = plt.subplot2grid((6, 4), (0, 1), colspan=1)
ax3 = plt.subplot2grid((6, 4), (2, 1), colspan=1)
ax5 = plt.subplot2grid((6, 4), (4, 1), colspan=1)

ax2 = plt.subplot2grid((6, 4), (0, 2), rowspan=2, colspan=2)
ax4 = plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
ax6 = plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
axlist = [ax1,ax2,ax3,ax4,ax5,ax6] 
plt.subplots_adjust(wspace=-0.4, hspace=1.1)

examples = [36, 200, 233 ]
nx, ny = (100, 100)
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
xv, yv = np.meshgrid(x, y)
cart = xv*1j + yv
pol = (np.rad2deg(np.angle(cart)))%360
mag = abs(cart)
color_circle = np.apply_along_axis(ziphusl, 2, np.dstack((pol, 100*np.ones_like(mag), 70*np.ones_like(mag))))
for example, ind in zip(examples, range(0,6,2)):
    #plt.subplot(3,2, ind+1)
    ax = axlist[ind+1]
    
    if use_cor:
        if ind == 0:
            ax.set_xlabel('PC1', fontsize=12);ax.set_ylabel('PC2',fontsize=12)
            ax.set_title(r"$R^2= $"+ str(np.round(da_sum_cor[example].values**2, 2)))
        else:
            ax.set_title(str(np.round(da_sum_cor[example].values**2, 2)))
    else:
        if ind == 0: 
            ax.set_xlabel('PC1', fontsize=12);ax.set_ylabel('PC2', fontsize=12)
            ax.set_title("Wt. Cov.="+ str(np.round(opponency_da[example].values, 2)))
        else:
            ax.set_title(str(np.round(opponency_da[example].values, 2)))
    #plt.title('R^2 = ' + str(np.round(da_sum_cor[example].values**2, 2)))
    ax.imshow(color_circle, interpolation='nearest')
    shift = 50
    scale = 40./np.max((coefs_da[example]**2).sum('chan')**0.5)
    rgb = coeffs_pol_rgb[:, example,...]
    
    ax.scatter((coefs_da[example, 0, ...]*scale)+shift, 
                (coefs_da[example, 1, ...]*scale)+shift,
                s=25, c=np.moveaxis(rgb.values.reshape(3, 25), 0, -1), edgecolors='k')
    [ax.spines[pos].set_visible(False) for pos in ['left','right','bottom','top']]

    ax.set_xticks([]);ax.set_yticks([])
    #plt.subplot(3,2, ind)
    ax = axlist[ind]
    ax.set_title(r'Filter: ' + str(example))

    ax.imshow(np.moveaxis(rgb.values, 0, -1), interpolation='nearest')

    ax.set_xticks([]);ax.set_yticks([])
    [ax.spines[pos].set_visible(False) for pos in ['left','right','bottom','top']]

    
#plt.tight_layout()
if use_cor:
    plt.savefig(top_dir + '/analysis/figures/images/early_layer/example_layer2_pc_vis_cor_title.pdf')
else:
    plt.savefig(top_dir + '/analysis/figures/images/early_layer/example_layer2_pc_vis_wt_cov.pdf')

#%%
plt.style.use(top_dir + '/poster/dean_poster.mplstyle')

opp_list = [spatial_opponency(netwtsd[layer]) for layer in layer_names]

m = len(opp_list)
n = 1

plt.figure(figsize=(2,8))

for layer, n in zip(opp_list,  range(m)):
    ax = plt.subplot(m, 1, n+1)
    #ax.annotate(str(layer.layer_label[0].values), [0.7,0.7], xycoords='axes fraction', fontsize=16)

    if n==0:
        ax.set_ylabel('Count', labelpad=4) 
        ax.set_xticks([0, 0.5 ,1])
        ax.set_xticklabels([])
    else:
        ax.set_xticks([0, 0.5 ,1])
        ax.set_xticklabels([])
        
    if n==m-1:
        ax.set_xticks([0,0.5,1])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_xlabel('Normalized\nWeight Covariance')
        
        
    ax.hist(layer, normed=0, bins=100, range=[-1,1])
    ax.set_xlim(-0.2,1)


plt.savefig(top_dir + '/analysis/figures/images/early_layer/deep_layers_covariance.pdf',
            bbox_inches='tight')
#%%
#plt.tight_layout()

##%%
#ravel_wts = [netwtsd[layer].values.reshape((netwtsd[layer].shape[0],) +
#             (np.product(netwtsd[layer].shape[1:]),)) 
#            for layer in layer_names]
one_pos = da.isel(x=5,y=5).squeeze()
layer_resp_list = [one_pos[:, one_pos.layer_label==layer] for layer in layer_names]
#
#num_lays = 7
#for resp, wts, layer_name in zip(layer_resp_list[:num_lays], ravel_wts, layer_names):
#    plt.figure(figsize=(3, 3))
#    plt.scatter(np.tril(np.corrcoef(resp.T),-1).ravel(), 
#                np.tril(np.corrcoef(wts), -1).ravel(), s=0.1)
#    r = (np.corrcoef(np.tril(np.corrcoef(resp.T),-1).ravel(), 
#                np.tril(np.corrcoef(wts), -1).ravel())[0,1])
#    plt.title('R = ' + str(np.round(r,2)) + ' ' + layer_name )
#    
#    plt.xlim(-1,1)
#    plt.ylim(-1,1)
#    plt.xlabel('Response Correlation')
#    plt.ylabel('Weights Correlation')

#%%
ravel_wts = [netwtsd[layer].values.reshape(netwtsd[layer].shape[:2] +
             (np.product(netwtsd[layer].shape[2:]),)) 
            for layer in layer_names]
one_pos = da.isel(x=5,y=5).squeeze()
the_input_names = ['norm1', 'relu2','relu3', 'relu4', 'pool5']
the_inputs = [one_pos[:, one_pos.layer_label==input_name] for input_name in the_input_names]
group_split = [1, 0 , 1, 1, 0]
wt_cors = []
resp_cors = []
for resp, wts, an_input, split in zip(layer_resp_list[1:6], ravel_wts[1:], the_inputs, group_split):
    if split:
        mid_wt = int(wts.shape[1])
        mid_unit = int(wts.shape[0]/2)
    else:
        mid_wt = int(wts.shape[0])
        mid_unit = int(wts.shape[0]/2)

    one_pos_resp = np.matmul(np.expand_dims(an_input[:, :mid_wt], 0), wts[:mid_unit, ...])
    resp_cors.append(np.array([np.corrcoef(unit.T) for unit in one_pos_resp]))
    wt_cors.append(np.array([np.corrcoef(unit.T) for unit in wts[:mid_unit, ...]]))
    
#%%

plt.style.use(top_dir + '/poster/dean_poster.mplstyle')

m = len(wt_cors)
n = 1
# We'll use two separate gridspecs to have different margins, hspace, etc
gs_top = plt.GridSpec(m, 1, top=0.95, left=0.4, hspace=0.3)
gs_base = plt.GridSpec(m, 1, hspace=0.3, left=0.4)
fig = plt.figure(figsize=(8,16))

# Top (unshared) axes
topax = fig.add_subplot(gs_top[0, :])
# The four shared axes
#ax = fig.add_subplot(gs_base[1, :]) # Need to create the first one to share...
#other_axes = [fig.add_subplot(gs_base[i,:], sharex=ax) for i in range(2, m)]
#bottom_axes = [ax] + other_axes

for wt_cor, resp_cor, n, layer_name in zip(wt_cors[:1], resp_cors,  range(m), layer_names[1:]):
    if n==0:
        ax = topax
        #ax.set_title(layer_name)
        ax.set_ylabel('Response Correlation')
        ax.set_xlabel('Weight Correlation')
        ax.set_xticks([-1,-0.5, 0,0.5,1])
        ax.set_yticks([-1,-0.5, 0,0.5,1])
        ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
        ax.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'])
        
    else:
        ax = bottom_axes[n-1]
        ax.set_title(layer_name)
        ax.set_xticks([-1,-0.5, 0,0.5,1])
        ax.set_yticks([-1,-0.5, 0,0.5,1])
        ax.set_xticklabels([])
        
        
    ax.set_aspect('equal')
    ax.scatter(np.tril(wt_cor, -1), np.tril(resp_cor, -1), s=.1)
    ax.set_xlim(-1,1);ax.set_ylim(-1,1)
plt.savefig(top_dir + '/analysis/figures/images/early_layer/response_wts_correlation.pdf',
            bbox_inches='tight')

##%%%
#plt.figure(figsize=(3,3))
#(c, x, y, Im) = plt.hist2d(np.tril(wt_cor, -1).ravel(), np.tril(resp_cor, -1).ravel(), bins=100)
##%%
#plt.style.use(top_dir + '/poster/dean_poster.mplstyle')
#   
#c_cond = c/np.sum(c, 0, keepdims=True)
#plt.imshow(np.flipud(np.log(c_cond)))
#%%
subsamp =1 
da = xr.open_dataset(top_dir + '/data/responses/bvlc_reference_caffenet_APC362_pix_width[32.0]_x_(74.0, 154.0, 21)_y_(74.0, 154.0, 21)_amp_None.nc')['resp']
net_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc'
da = xr.open_dataset(top_dir + '/data/responses/'+net_name)['resp'].squeeze()

#da = da[:,0,:,:,:]
#da.dims
da = da.transpose('unit','shapes', 'x', 'y')
da = da[::subsamp, ...] #subsample
da = da.load()
da = da - da[:, 0, :, :] #subtract off baseline
da = da[:, 1:, ...] #get rid of baseline shape   
from scipy.stats import kurtosis
#%%
def ti_av_cov(da):
    dims = da.coords.dims
    #get the da in the right shape
    if ('x' in dims) and ('y' in dims):
        da = da.transpose('unit','shapes', 'x', 'y')
    elif ('x' in dims):
        da = da.transpose('unit', 'shapes', 'x')
    elif ('y' in dims):
        da = da.transpose('unit', 'shapes', 'y')
        
    #some data to store
    ti = np.zeros(np.shape(da)[0])
    dens = np.zeros(np.shape(da)[0])
    nums = np.zeros(np.shape(da)[0])
    tot_vars = np.zeros(np.shape(da)[0])
    kurt_shapes = np.zeros(np.shape(da)[0])
    kurt_x =  np.zeros(np.shape(da)[0])

    for i, unit_resp in enumerate(da):
        if len(unit_resp.shape)>2:
            #unwrap spatial
            unit_resp = unit_resp.values.reshape(unit_resp.shape[0], unit_resp.shape[1]*unit_resp.shape[2])   
        else:
            unit_resp = unit_resp.values
        unit_resp = unit_resp.astype(np.float64)
        unit_resp = unit_resp - np.mean(unit_resp, 0, keepdims=True, dtype=np.float64)
 

        cov = np.dot(unit_resp.T, unit_resp)
        cov[np.diag_indices_from(cov)] = 0
        numerator = np.sum(np.triu(cov))

        vlength = np.linalg.norm(unit_resp, axis=0, keepdims=True)
        max_cov = np.outer(vlength.T, vlength)
        max_cov[np.diag_indices_from(max_cov)] = 0
        denominator= np.sum(np.triu(max_cov))

        kurt_shapes[i] = kurtosis(np.sum(unit_resp**2, 1))
        kurt_x[i] = kurtosis(np.sum(unit_resp**2, 0))
        den = np.sum(max_cov)
        num = np.sum(cov)
        dens[i] = den
        nums[i] = num
        tot_vars[i] = np.sum(unit_resp**2)
        if den!=0 and num!=0:
            ti[i] = num/den 
    return ti, kurt_shapes, kurt_x, dens, nums, tot_vars 
def wts_av_cov(da, inclmean=True, rtrn_space=False):
    dims = da.coords.dims
    #get the da in the right shape
    if ('x' in dims) and ('y' in dims):
        da = da.transpose('unit','shapes', 'x', 'y')
        ti_spc = np.zeros((np.shape(da)[0], np.shape(da)[2], np.shape(da)[3]))
    elif ('x' in dims):
        da = da.transpose('unit', 'shapes', 'x')
        ti_spc = np.zeros((np.shape(da)[0], np.shape(da)[2]))
    elif ('y' in dims):
        da = da.transpose('unit', 'shapes', 'y')
        ti_spc = np.zeros((np.shape(da)[0], np.shape(da)[2]))
        
    #some data to store
    ti = np.zeros(np.shape(da)[0])
    
    
    dens = np.zeros(np.shape(da)[0])
    nums = np.zeros(np.shape(da)[0])
    tot_vars = np.zeros(np.shape(da)[0])
    kurt_shapes = np.zeros(np.shape(da)[0])
    kurt_x =  np.zeros(np.shape(da)[0])

    for i, unit_resp in enumerate(da):
        if len(unit_resp.shape)>2:
            #unwrap spatial
            unit_resp = unit_resp.values.reshape(unit_resp.shape[0], unit_resp.shape[1]*unit_resp.shape[2])   
        else:
            unit_resp = unit_resp.values
        unit_resp = unit_resp.astype(np.float64)
        if not inclmean:
            unit_resp = unit_resp - np.mean(unit_resp, 0, keepdims=True, dtype=np.float64)
 

        cov = np.dot(unit_resp.T, unit_resp)
        cov[np.diag_indices_from(cov)] = 0
        cov_spc = cov.sum(0)
        numerator = np.sum(np.triu(cov))

        vlength = np.linalg.norm(unit_resp, axis=0, keepdims=True)
        max_cov = np.outer(vlength.T, vlength)
        max_cov[np.diag_indices_from(max_cov)] = 0
        max_cov_spc = max_cov.sum(0)
        denominator= np.sum(np.triu(max_cov))
        
        
        #trying to output correctly the shape of normalized covariance in a map.
        
        den = np.sum(max_cov)
        num = np.sum(cov)
        dens[i] = den
        nums[i] = num
        tot_vars[i] = np.sum(unit_resp**2)
        if den!=0 and num!=0:
            ti[i] = num/den
            ti_spc[i] = np.reshape(cov_spc / max_cov_spc, np.shape(da)[2:])
        
        
    if rtrn_space:
        return ti_spc
    else:
        return ti
#%%
ti_yx, kurt_shapes_yx, kurt_yx, dens, nums, tot_vars_yx = ti_av_cov(da[:, :, :, :])
#%%
import pandas as pd
ti_by_layer = []
layer_labels = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6']
for layer, layer_name in zip(netwts, layer_labels):
    print(layer[1].shape)
    if len(layer[1].shape)>2:
        _ = xr.DataArray(layer[1], dims=['unit', 'shapes', 'x', 'y'])
        ti = wts_av_cov(_)
        print(len(ti))
        ti_by_layer.append(ti)
wt_cov = np.concatenate(ti_by_layer)

non_k_var = (kurt_shapes_yx<42) * (kurt_shapes_yx>2) * (tot_vars_yx>0) 
keys = ['layer_label', 'unit']
coord = [da.coords[key].values for key in keys]
index = pd.MultiIndex.from_arrays(coord, names=keys)
resp = pd.DataFrame(np.hstack([ti_yx,]), index=index, columns=['ti',])
layersbyunit = [[name,]*layer_wts[1].shape[0] for name, layer_wts in zip(layer_labels, netwts)]
keys = ['layer_label',]
index = pd.MultiIndex.from_arrays([np.concatenate(layersbyunit),], names=keys)
#%%
wts = pd.DataFrame(np.vstack([wt_cov,]).T, index=index, columns=['wts_cov',])
n_plots = len(layer_labels[1:])
plt.figure(figsize=(12,3))

for i, layer in enumerate(layer_labels[1:]):
    plt.subplot(1, n_plots, i+1)
    x = wts.loc[layer]['wts_cov'].values
    y = np.squeeze(resp.loc[layer].values)
    if i<4:
        s=4
    else:
        s=1
    plt.scatter(x, y, s=s, color='k', edgecolors='none')
    #plt.semilogx()
    plt.xlim(-0.1,1.02);plt.ylim(-0.1,1.01);
    if i==0:
        plt.xlabel('Weight Covariance'); plt.ylabel('T.I.', rotation=0, va='center',ha='right', labelpad=15)
    if layer == 'conv2':
        plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['0','','0.5','','1'])
        plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['0','','0.5','','1'])
        plt.title(layer + '\nr = ' + str(np.round(np.corrcoef(x,y)[0,1], 2)))

    else:
        plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['','','','',''])
        plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['','','','',''])
        plt.title(layer + '\n' + str(np.round(np.corrcoef(x,y)[0,1], 2)))
    plt.tight_layout()
    plt.grid()
plt.savefig(top_dir + '/analysis/figures/images/early_layer/34_pix_wt_cov_vs_TI.pdf', bbox_inches='tight')

#%%
the_input_names = ['norm1', 'relu2','relu3', 'relu4', 'pool5']
layer_labels = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6']

group_split = [1, 0 , 1, 1, 0]
chan_wt_ti_prev_r = []
for layer_cur, layer_prev, split in zip(layer_labels[1:], the_input_names, group_split):
    ti_prev = np.squeeze(resp.loc[layer_prev].values)
    chan_wt_cur = (netwtsd[layer_cur]).sum(['x','y'])
    if split:
        mid_unit_in = int(chan_wt_cur.shape[1])
        mid_unit_out = int(chan_wt_cur.shape[0]/2)
    else:
        mid_unit_in = int(chan_wt_cur.shape[1])
        mid_unit_out = int(chan_wt_cur.shape[0])
        
    print(ti_prev.shape)
    print(chan_wt_cur.shape)
    _ = np.array([np.corrcoef(ti_prev[:mid_unit_in], chan_wt)[0,1] 
                for chan_wt in chan_wt_cur[:mid_unit_out,]])
    print('cor_ out :' + str(_.shape))
    chan_wt_ti_prev_r.append(_)
    
n_plots = len(layer_labels[1:])
#%%
plt.figure(figsize=(4, 12))
for i, layer in enumerate(layer_labels[1:]):
    plt.subplot(n_plots, 1, i+1)
    x = chan_wt_ti_prev_r[i]
    y = np.squeeze(resp.loc[layer].values[:x.shape[0]])
    plt.scatter(x, y, s=1, color='k', edgecolors='none')
    #plt.semilogx()
    plt.xlim(-.5,.5);plt.ylim(0,1);plt.xlabel('R Chan Sum vs TI Input'); plt.ylabel('TI')
    plt.title(layer + ' r= ' + str(np.round(np.corrcoef(x, y)[0,1],2)))
    plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/early_layer/34_pix_chan_wt_sum_ti_r_vs_ti.pdf', bbox_inches='tight')

#%%
plt.figure(figsize=(4, 12))
for i, layer in enumerate(layer_labels[1:]):
    plt.subplot(n_plots, 1, i+1)
    x = chan_wt_ti_prev_r[i]
    y = wts.loc[layer]['wts_cov'].values[:x.shape[0]]
    plt.scatter(x, y, s=1, color='k', edgecolors='none')
    #plt.semilogx()
    plt.xlim(-.5,.5);plt.ylim(0,1);plt.xlabel('R Chan Sum vs TI Input'); plt.ylabel('Wts Cov')
    plt.title(layer + ' r= ' + str(np.round(np.corrcoef(x, y)[0,1],2)))
    plt.tight_layout()
plt.savefig(top_dir + '/analysis/figures/images/early_layer/34_pix_chan_wt_sum_ti_r_vs_wt_cov.pdf', bbox_inches='tight')
