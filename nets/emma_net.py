# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:03:59 2017

@author: deanpospisil
"""

import numpy as np
import scipy as sc
from scipy import signal
import pickle as pk
import os, sys
import matplotlib.pyplot as plt
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import d_img_process as imp
import caffe_net_response as cf
from itertools import product


def translateByPixels(img,x,y):
    x = int(np.round(x))
    y = int(np.round(y))
    newImg= np.zeros(np.shape(img))
    nrows= np.size(img,0)
    ncols= np.size(img,1)
    r , c = np.meshgrid( range(nrows), range(ncols) );

    newrow = r-y
    newcol = c+x

    valid = (newrow<nrows) & (newcol<ncols) & (newcol>=0) & (newrow>=0)
    r =  r[valid]
    c =  c[valid]
    newrow = newrow[valid]
    newcol = newcol[valid]

    newImg[newrow,newcol] = img[r,c]

    return newImg


goforit=False 
if 'a' not in locals() or goforit:
    with open(top_dir + 'nets/netwts.p', 'rb') as f:    
        try:
            a = pk.load(f, encoding='latin1')
        except:
            a = pk.load(f)
            
img_dir = top_dir + 'images/baseimgs/PC370/'

    
inds = range(48)
kernels_lay1 = a[0][1][inds]
bias_lay1 = a[0][0][inds]

inds = range(128)
kernels_lay2 = a[1][1][inds]
bias_lay2 = a[1][0][inds]

image = np.zeros((3, 227, 227))
image = np.random.randn(3, 227, 227)
image = np.repeat(np.expand_dims(np.load(img_dir + '0.npy'), 0), 3, axis=0)

img_n_pix = 227
shape_ids = range(0, 370)
center_image = round(img_n_pix/2.)
x = (center_image-10, center_image+10, 21)
y = (center_image, center_image, 1)

stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                                                                x=x, y=y)
stack, stack_desc = imp.load_npy_img_dirs_into_stack(img_dir)
trans_stack = imp.imgStackTransform(stim_trans_cart_dict, stack)

#%%

def stride_conv(kernels, image, stride=1):
    cv_channels = np.zeros((kernels.shape[0],) 
                            + (int(np.ceil(np.shape(image)[1]/stride)),)
                            + (int(np.ceil(np.shape(image)[2]/stride)),))
    cv = np.zeros((image.shape[0],) + cv_channels.shape[1:])
    
    for i, kernel in enumerate(kernels):
        cv = np.array([signal.fftconvolve(channel_image, channel_kern, 'same') 
                        for channel_kern, channel_image 
                        in zip(kernel, image)])
        cv = cv[:, ::stride, ::stride]
        cv = np.sum(cv, 0)    
        cv_channels[i] = cv
    return cv_channels
def max_pool(stride, neighborhood, cvds):
    max_pools = []
    cart = list(product(range(0, cvds.shape[1], stride), 
                              range(0, cvds.shape[2], stride)))
    for cvd in cvds:
        max_pool = np.array([np.max(cvd[coord[0]:coord[0]+neighborhood, 
                                        coord[1]:coord[1]+neighborhood]) 
                for coord in cart])
        #max_pools.append(max_pool)
        max_pools.append(max_pool.reshape((int(max_pool.shape[0]**0.5),)*2))
    max_pools = np.array(max_pools)
    return max_pools

all_ims = []
for image in trans_stack:
    image = np.repeat(np.expand_dims(image, 0), 3, axis=0)
    stride = 4
    cvds = stride_conv(kernels_lay1, image, stride)
    cvds = cvds + np.reshape(bias_lay1, (bias_lay1.shape[0], 1, 1))
    
    #rectify
    cvds[cvds<0] = 0
    
    #pool
    stride = 2
    neighborhood = 3
    max_pools = max_pool(stride, neighborhood, cvds)
    
    #norm
    local_size = 5
    alpha = 0.0001
    beta = 0.75
    sum_conv = np.ones(local_size)
    max_pools_ravel = np.reshape(max_pools, (np.shape(max_pools)[0], 
                                             np.product(np.shape(max_pools)[1:])))
    wndwd_sum = [np.convolve(np.pad(sec**2, local_size, 'constant', constant_values=0),
                             sum_conv, 'same') 
                             for sec in max_pools_ravel.T]
    wndwd_sum = np.array(wndwd_sum)
    wndwd_sum = wndwd_sum[:, local_size:-local_size]
    wndwd_sum = (1+(alpha/local_size)*wndwd_sum)**beta
    
    normd = max_pools_ravel / (wndwd_sum.T+0.1)
    normd = normd.reshape((np.shape(max_pools)[0],)
                            + (int(max_pools_ravel.shape[1]**0.5),)*2)
    
    
    #cvds_2 = stride_conv(kernels_lay2, normd, stride=1)
    #cvds_2 = cvds_2 + np.reshape(bias_lay2, (bias_lay2.shape[0], 1, 1))

    cvds_2 = np.array([np.dot(kernel.ravel(), normd[:, 12:17, 12:17].ravel()) 
                        for kernel in kernels_lay2]) + bias_lay2
    all_ims.append(cvds_2)
all_ims = np.array(all_ims)
#%%
import xarray as xr
#all_ims_center = all_ims[:,:, 15, 15]
all_ims_center = all_ims
#get the dimensions of the stimuli in order.
dims = tuple([len( stim_trans_dict[key]) for key in stim_trans_dict]) + tuple([all_ims_center.shape[1],])
#reshape into net_resp_xray
#this working is dependent on cartesian producing A type cartesian
#(last index element changes fastest)
net_resp_xray_vals = np.reshape(all_ims_center, dims)
net_dims = [key for key in stim_trans_dict] + ['unit',]
net_coords =[stim_trans_dict[key] for key in stim_trans_dict] + [range(dims[-1])]
da = xr.DataArray(net_resp_xray_vals, coords=net_coords, dims=net_dims)