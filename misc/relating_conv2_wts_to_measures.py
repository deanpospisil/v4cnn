# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:39:32 2017

@author: deanpospisil
"""

import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr;import pandas as pd
import apc_model_fit as ac
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp
import d_net_analysis as dn
from sklearn.neighbors import KernelDensity
import caffe_net_response as cf


def vis_square(ax, data, padsize=0, padval=0):
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    ax.set_xticks([]);ax.set_yticks([])
    return data
    
def plot_resp_on_shapes(ax, imgStack, resp, image_square=19):
    resp_sc = resp.values
    resp_sc -= np.mean(resp_sc)
    max_dist = np.max(np.abs(resp_sc))
    imgStack = imgStack*resp_sc.reshape(362,1,1)
    #sort images
    sortStack = imgStack[list(reversed(np.argsort(resp_sc))),:,:]
    sortStack = np.array([imp.centeredCrop(img, 64, 64) for img in sortStack])
    im = vis_square(ax, sortStack[0:image_square**2])
    return im

def open_cnn_analysis(fn,layer_label):
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
    apc = dn.ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                  dmod.chunk({}), fit_over_dims=None, 
                                    prov_commit=False)**2.
    k_apc = list(dn.kurtosis(v4_resp_apc).values)
    k_ti = list(dn.kurtosis(v4_resp_ti.mean('x')).values)

    keys = ['layer_label', 'unit']
    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(ti)),
                                       np.arange(len(ti))], names=keys)
    v4pdti  = pd.DataFrame(np.array([ti, k_ti]).T, index=index, 
                           columns=['ti_av_cov', 'k'])

    index = pd.MultiIndex.from_arrays([np.array(['v4']*len(apc)), 
                                       np.arange(len(apc))], 
                                       names=keys)
    coords_to_take= ['cur_mean', 'cur_sd', 'or_mean', 'or_sd', 'models']
    apc_coords = [apc.coords[coord].values for coord in coords_to_take]
    v4pdapc  = pd.DataFrame(np.array([apc.values,] + apc_coords + [k_apc,]).T, 
               index=index, 
               columns=['apc', ] + coords_to_take + [ 'k',])
    v4 = pd.concat([v4pdti, v4pdapc])
    return v4
#%%
#shape image set up
img_n_pix = 100
max_pix_width = [80,]
s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(-1, 370); center_image = round(img_n_pix/2)
x = (center_image, center_image, 1);
y = (center_image, center_image, 1)
stim_trans_cart_dict, _ = cf.stim_trans_generator(shapes=shape_ids, scale=scale, 
                                                  x=x, y=y)
#plt.figure(figsize=(12,24));
center = np.ceil(img_n_pix/2.)
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict, 
                                                        base_stack, 
                                                        npixels=img_n_pix))
no_blank_image = trans_img_stack[1:]
a = np.hstack((range(14), range(18, 318)));a = np.hstack((a, range(322, 370)))
no_blank_image = no_blank_image[a]/255.
#%%

goforit = 0
#loading up all needed data
if 'cnn_an' not in locals() or goforit:
    v4_name = 'V4_362PC2001'
    v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    file = open(top_dir + 'data/responses/v4_apc_109_neural_labels.txt', 'r')
    wyeth_labels = [label.split(' ')[-1] for label in 
                file.read().split('\n') if len(label)>0]
    v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
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
            v4_resp_ti_null[unit, x, not_null] = np.random.permutation(
                                                 v4_resp_ti[unit, x, not_null].values)
    
    v4_resp_apc = v4_resp_apc.transpose('shapes','unit')
    for unit in range(len(v4_resp_apc_null.coords['unit'])):
        v4_resp_apc_null[:, unit] = np.random.permutation(v4_resp_apc[:, unit].values)

    null_v4 = process_V4(v4_resp_apc_null, v4_resp_ti_null, dmod)
    
    cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',]
    
    da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
    da = da.sel(unit=slice(0, None, 1)).squeeze()
    middle = np.round(len(da.coords['x'])/2.).astype(int)
    da_0 = da.sel(x=da.coords['x'][middle])
    indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
    layer_label = [da.coords['layer_label'].values[index] for index in sorted(indexes)]
       
    fns = [
    'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
    'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_null_analysis.p'
    ]

    alt = pd.concat([open_cnn_analysis(fns[0], layer_label)[-1], alt_v4], axis=0)
    init = open_cnn_analysis(fns[1], layer_label)[-1]
    shuf = open_cnn_analysis(fns[2], layer_label)[-1]
    null = pd.concat([open_cnn_analysis(fns[3], layer_label)[-1], null_v4], axis=0)
    cnn_an = pd.concat([alt, null, init, shuf ], 
              axis=0, keys=['resp', 's. resp', 'init. net', 's. layer wts'], names=['cond','layer_label','unit'])

#%%
goforit=True    
if 'a' not in locals() or goforit:
    with open(top_dir + 'nets/netwts.p', 'rb') as f:    
        try:
            a = pk.load(f, encoding='latin1')
        except:
            a = pk.load(f)
inds = range(256)
wts = a[1][1][inds]

ti_conv2 = cnn_an.loc['resp']['ti_av_cov'].loc['conv2'].values[inds]
rf = np.sum(wts**2, 1)
rf_max = np.max(rf, (1, 2))
rf_power = np.sum(rf, (1, 2))
rf_conc = rf_max/rf_power

plt.scatter(rf_conc, ti_conv2)
plt.xlabel('max percent of total')
plt.ylabel('TI')

plt.figure()
inds = range(128)
cnn_an.loc['resp']['ti_av_cov'].loc['conv2'].iloc[inds].plot(kind='hist', histtype='step')
inds = range(128,256)
cnn_an.loc['resp']['ti_av_cov'].loc['conv2'].iloc[inds].plot(kind='hist', histtype='step')

plt.figure()
plt.scatter(cnn_an.loc['resp']['k']['conv2'], ti_conv2)
plt.figure()
plt.scatter(cnn_an.loc['resp']['apc']['conv2']**0.5, ti_conv2)

plt.figure()
plt.scatter(cnn_an.loc['resp']['cur_mean']['conv2']**0.5, ti_conv2)



#%%
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
def coef_var(a):
    mu = a.mean()
    sig = a.std()
    return 1./(((sig/mu)**2)+1)
    
def sinusoid_weights_test(orientation_of_inputs, weights_on_outputs, freq=2):
    lyr_2_prd_df = xr.DataArray([np.cos(freq*orientation_of_inputs.values), 
                                 np.sin(freq*orientation_of_inputs.values)], 
                                dims=['p','l1'])
    lyr_2_prd_df_nrm = lyr_2_prd_df / (lyr_2_prd_df**2).sum('l1')**0.5
    
    fits = (lyr_2_prd_df_nrm*lyr_2_wts_df).sum('l1').squeeze()
    lyr_2_wts_df_hat = (fits * lyr_2_prd_df_nrm).sum('p')

    lyr_2_wts_df_hat_nrm = lyr_2_wts_df_hat / (lyr_2_wts_df_hat**2).sum(['l1','r','c'])**0.5
    lyr_2_wts_df_nrm = lyr_2_wts_df / (lyr_2_wts_df**2).sum(['l1','r','c'])**0.5
    cor = (lyr_2_wts_df_hat_nrm * lyr_2_wts_df_nrm).sum(['l1','r','c'])
    return cor

def orientation_power(filt):
    a_filter_unwrap = filt.reshape(np.product(filt.shape[:-1]), 3)
    u, s, v = np.linalg.svd(a_filter_unwrap, full_matrices=False)
    val_map.append(np.dot(a_filter_unwrap, v[0,:]).reshape(np.shape(a_filter)[:-1]))
    val_map = np.array(val_map)
    upsampled_fft_amplitude = np.abs(np.fft.fft2(filt, 
                            s=np.array(np.shape(first_layer_weights_grey_scale)[1:])*sample_rate_mult))
    
    first_layer_weights = np.array([im for im in ims])
    first_layer_weights_grey_scale = np.sum(ims, 1)[:lyr_1_grp_sze, ...]
    first_layer_weights_grey_scale -= np.mean(first_layer_weights_grey_scale, axis =(1,2), keepdims=True)
    upsampled_fft_amplitude = np.abs(np.fft.fft2(first_layer_weights_grey_scale, 
                            s=np.array(np.shape(first_layer_weights_grey_scale)[1:])*sample_rate_mult))

    polar = [img2polar(np.fft.fftshift(a_filter), [55,55], 55, phase_width=360)
                        for a_filter in upsampled_fft_amplitude]
    polar_amp_kurtosis = np.array([coef_var(polar_filter.sum(0)) for polar_filter in polar])
    prfrd_ori_deg = np.array([np.argmax(polar_filter.sum(0)) for polar_filter in polar])
    prfrd_ori_rad = np.deg2rad(prfrd_ori_deg)
    prfrd_ori_rad_wrp = prfrd_ori_rad%np.pi

if not 'afile' in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            afile = pk.load(f, encoding='latin1')
        except:
            afile = pk.load(f)
wts = np.transpose(a[0][1], (0,2,3,1))

val_map = []
s_list = []
for a_filter in wts:
    a_filter_unwrap = a_filter.reshape(np.product(a_filter.shape[:-1]),3)
    u,s, v = np.linalg.svd(a_filter_unwrap, full_matrices=False)
    s_list.append(s[0]/s.sum())
    val_map.append(np.dot(a_filter_unwrap, v[0,:]).reshape(np.shape(a_filter)[:-1]))
val_map = np.array(val_map)
           
layer = 0
sample_rate_mult = 10
ims = afile[layer][1]

lyr_1_grp_sze = 48
lyr_2_grp_sze = 128

first_layer_weights = np.array([im for im in ims])
first_layer_weights_grey_scale = val_map
first_layer_weights_grey_scale -= np.mean(first_layer_weights_grey_scale, axis =(1,2), keepdims=True)
upsampled_fft_amplitude = np.abs(np.fft.fft2(first_layer_weights_grey_scale, 
                        s=np.array(np.shape(first_layer_weights_grey_scale)[1:])*sample_rate_mult))

polar = [img2polar(np.fft.fftshift(a_filter), [55,55], 55, phase_width=360)
                    for a_filter in upsampled_fft_amplitude]
polar_amp_kurtosis = np.array([coef_var(polar_filter.sum(0)) for polar_filter in polar])
prfrd_ori_deg = np.array([np.argmax(polar_filter.sum(0)) for polar_filter in polar])
prfrd_ori_rad = np.deg2rad(prfrd_ori_deg)
prfrd_ori_rad_wrp = prfrd_ori_rad%np.pi

power_concentration = upsampled_fft_amplitude.max((1,2)) / upsampled_fft_amplitude.sum((1,2))
top_pwr_cncntrtn_ind = polar_amp_kurtosis<np.percentile(polar_amp_kurtosis, 80)
peak_ori = prfrd_ori_rad_wrp
bandwidth = power_concentration


lay1_1 = np.array(peak_ori[:48])
lay1_2 = peak_ori[48:]
lay2 = a[1][1]
lay2 = lay2.reshape(256,48,25)
lay2_1 = lay2[:128]
#lay2_1 -= lay2_1.mean(1, keepdims=True)
lay2_2 = lay2[128:]
#lay2_2 -= lay2_2.mean(1, keepdims=True)
#%%
freq=2
#our predictors are a sinusoidal function the preferred orientation of the prior layer
A = np.vstack([np.cos(lay1_1*freq), np.sin(lay1_1*freq), np.ones(len(lay1_1))]).T
#we individually fit our predictors to each 'pixel' 
lay2_1_ms = np.array([a_filt-np.mean(a_filt,1, keepdims=True) for a_filt in lay2_1])
fit = [np.linalg.lstsq(A, a_filt) for a_filt in lay2_1_ms]
frac_var_tot1 = [1 - a_fit[1].sum()/np.sum(a_filt**2) for a_fit, a_filt in zip(fit, lay2_1_ms)]
#%%
inds = range(128)
plt.scatter(frac_var_tot1, cnn_an.loc['resp']['ti_av_cov'].loc['conv2'].iloc[inds])
plt.xlabel('Cross-Ori');
plt.ylabel('TI')
#%%
fig, axes = plt.subplots(nrows=11, ncols=12, figsize=(10,10))

lay2 = a[1][1]
wt_pwr2_1 = np.sum(lay2[:128]**2, 1, keepdims=False)
wt_pwr2_1 = wt_pwr2_1/np.sum(wt_pwr2_1, (1,2), keepdims=True)

wt_pwr2_2 = np.sum(lay2[128:]**2, 1, keepdims=False)
wt_pwr2_2 = wt_pwr2_2/np.sum(wt_pwr2_2, (1,2), keepdims=True)

for ax in axes.flat:
    ax.set_yticks([])
    ax.set_xticks([])
sort_ti_ind  = np.argsort(ti_conv2[:128])
for a_filter, ax, ti in zip(wt_pwr2_1[sort_ti_ind], axes.flat, ti_conv2[sort_ti_ind]):
    im = ax.imshow(a_filter, interpolation='none', cmap=cm.viridis_r, vmin=0, vmax=0.2)
    ax.set_title(str(np.round(ti,2)))
    
plt.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Percent Variance')
#%%
fns = [
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_null_analysis.p'
]
layer = 'conv1'
a = cnn_an.loc['resp']['ti_av_cov'].loc[layer].index.values
ti = cnn_an.loc['resp']['ti_av_cov'].loc[layer].values

sort_ti_ind = np.argsort(ti)
rf = open_cnn_analysis(fns[0], layer_label)[1][a][sort_ti_ind]
cor = open_cnn_analysis(fns[0], layer_label)[0][a][sort_ti_ind]
rf = open_cnn_analysis(fns[0], layer_label)[1][a]
cor = open_cnn_analysis(fns[0], layer_label)[0][a]

ind = 20
cor[ind,:].plot()
(rf[ind,:]/rf[ind,:].max()).plot()
print(ti[sort_ti_ind][ind])
print(cor[ind,:].coords['layer_unit'].values)

#%%
s_list = []
for a_filter in lay2:
    s = np.linalg.svd(a_filter, compute_uv=False)
    s_list.append(s[0].sum()/s.sum())
plt.scatter(s_list, ti_conv2)
plt.xlabel('1st pc')
plt.ylabel('ti')
plt.xlim(0,1)
plt.ylim(0,1)

#%%
layers = ['fc6', 'fc7', 'fc8']
the_best_v4like = []
for layer in layers:
    tmp = cnn_an.loc['resp'].drop('v4', level='layer_label').loc[layer]
    comb_score = tmp['apc'] + tmp['ti_av_cov']
    the_best_v4like.append(list(comb_score.sort_values(inplace=False)[-10:-1].index.values))
the_best_v4like

#%%
