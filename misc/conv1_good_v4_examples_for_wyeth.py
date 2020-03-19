#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:46:42 2017

@author: dean
"""

import numpy as np
import os
import sys

#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')
sys.path.append(top_dir +'/nets')

import d_img_process as imp
import d_misc as dm
import d_curve as dc
import scipy.io as  l
import matplotlib.pyplot as plt
import d_curve as dc
s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)
import caffe_net_response as cf
import d_img_process as imp
import pandas as pd
import pickle as pk
import xarray as xr
def open_cnn_analysis(fn):
    try:
        an=pk.load(open(fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(fn,'rb'))
    fvx = an[0].sel(concat_dim='r')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    return fvx, rf, cnn
fns = [
'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
]

data_dir = '/loc6tb/'
results_dir = data_dir + '/data/an_results/'
cnn_an = open_cnn_analysis(results_dir +  fns[0])[-1]

cnn_names =['bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',]
da = xr.open_dataset(data_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
da = da.sel(unit=slice(0, None, 1)).squeeze()
middle = np.round(len(da.coords['x'])/2.).astype(int) -1
da_0 = da.sel(x=da.coords['x'][middle])
with open(top_dir + 'nets/netwts.p', 'rb') as f:    
    try:
        a = pk.load(f, encoding='latin1')
    except:
        a = pk.load(f)
wts = a[0][1]
fn = data_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp']
#%%
import apc_model_fit as ac
conv1 = da_0[:, :95]
apc_fit= ac.cor_resp_to_model(conv1.chunk({'shapes': 370}), 
                                  dmod.chunk({}), 
                                  fit_over_dims=None, 
                                  prov_commit=False)

#%%
corr = apc_fit.max()
best_ind = apc_fit.argsort()[-1]
mod = apc_fit.coords['models'][int(best_ind)].values
r = conv1[:, int(best_ind)]
m = dmod[:, int(mod)]
r = r.reindex_like(m)

print(np.corrcoef(r, m))
print(corr.values)
#%%
img_n_pix = 227
max_pix_width = [ 32.,]

mat = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')
s = np.array(mat['shapes'][0])
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
#scale = None
shape_ids = range(-1, 370)
center_image = round(img_n_pix/2.)
#y = (center_image-80, center_image+80, 21)
x = (center_image, center_image, 1)
#x = (center_image-80, center_image+80, 21)
y = (center_image, center_image, 1)

stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                                                                scale=scale,
                                                                x=x,
                                                                y=y)




#%%
n_best = 1
sort_ind = apc_fit.loc['conv1']['apc'].argsort().values[-n_best]
r = cnn_an.loc['conv1']['apc'][sort_ind]**0.5

b_resp = da_0[:, sort_ind]
b_resp = da_0.sel(unit=85)
b_model_params = (np.round(cnn_an['cur_mean'][sort_ind], 2), 
           np.round(np.rad2deg(cnn_an['or_mean'][sort_ind]),2))
b_weights = wts[sort_ind]

model_ind = cnn_an['models'][sort_ind]
model_ind = 63947
model_resp = dmod.sel(models=model_ind)
b_resp_s = b_resp.reindex_like(model_resp)

print(np.corrcoef(b_resp_s, model_resp)[0,1])
#%%
plt.figure(figsize=(5,20))
center = 114
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict, 
                                                        base_stack, npixels=img_n_pix))

n_top = 10
plt.subplot(n_top + 2, 1, 1)
filt = b_weights - b_weights.min()
filt = filt/filt.max()
plt.imshow(filt.T)
plt.xticks([]);plt.yticks([])
plt.title('r='+str(np.round(r,2))+' curv. ' + str(b_model_params[0])+ 
          ' ori. ' + str(b_model_params[1]))
plt.subplot(n_top + 2, 1, 2)
plt.plot(b_resp)
for i, top_shapes in enumerate(b_resp.argsort()[::-1].values[:n_top]):
    #plot smallest and largest shape
    plt.subplot(n_top + 2, 1, i+3)
    plt.imshow(trans_img_stack[top_shapes], cmap=plt.cm.Greys_r)
    #I take the 27th row 27th column conv1 unit (index from 0) at a stride of 4
    #4*27 shifts (0 has no shift) = 108
    rectangle = plt.Rectangle((108, 108), 11, 11, fill=False, edgecolor='r')
    plt.gca().add_patch(rectangle)
    plt.xticks([]);plt.yticks([])
    plt.xlim([80,140]);plt.ylim([80,140])

plt.savefig(top_dir + 'analysis/figures/images/example.jpeg')
#%%
net = '/home/dean/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet'
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
deploy='deploy_fixing_relu_saved.prototxt'

import caffe
caffe.set_mode_gpu()
net = caffe.Net(ann_dir + deploy, ann_dir + 'bvlc_reference_caffenet' + '.caffemodel', caffe.TEST)
#%%
image = np.zeros((1,227,227))
image[:, 108, 108] = 1
image[:, 109, 118] = 1

new_filt = np.zeros((3,11,11))
new_filt[:, 0, 0] = 1
new_filt[:, 1, 10] = 1
net.params['conv1'][0].data[0] = new_filt
net.params['conv1'][1].data[0] = 0

resp = cf.net_imgstack_response(net, image, only_middle_conv=True, record_up_to_layer=None)

print(resp[0,0])
#the 108th row and colum of the image corresponds to the 0th row and column of the filter 