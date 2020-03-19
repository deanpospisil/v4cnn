#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:18:43 2017

@author: dean
"""


# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict as ordDict

import os
import sys
import warnings

#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')

sys.path.append('/home/dean/caffe/python')

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys

import caffe
import os

def norm_cov(x, subtract_mean=True):
    
    #if nxm the get cov nxn
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 0, keepdims=True)
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    vnrm = np.linalg.norm(x, axis=0, keepdims=True)
    denominator = np.sum(np.multiply(vnrm.T, vnrm)[diag_inds]) 
    norm_cov = numerator/denominator
    
    return norm_cov
    
caffe.set_mode_gpu()
caffe_root = '/home/dean/caffe/'
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ann_fn = 'bvlc_reference_caffenet'

caffe.set_mode_gpu()

net = caffe.Net(ann_dir + 'deploy_fixing_relu_saved.prototxt', 
                ann_dir + ann_fn + '.caffemodel', 
                caffe.TEST)


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

img_dir = '/loc6tb/data/images/ILSVRC2012_img_val/'
img_names = os.listdir(img_dir)
img_names = [img_name for img_name in img_names if 'JPEG' in img_name]
#%%
layer_list = list(net.blobs.iteritems())[1:-6]
n_images = 20
n_layers = len(layer_list)
nc_array = np.zeros((n_images, n_layers))
for img_ind in range(n_images):
    image = caffe.io.load_image(img_dir + img_names[img_ind])
    
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    
    output = net.forward()
    print(img_ind)
    
    # for each layer, show the output shape
    layer_ind = 0
    for layer_name, blob in layer_list:
        c = blob.data.shape[-1]
        cent = int(c/2)
        mid_resp = np.squeeze(blob.data[0, :, cent-2:cent+2, cent-2:cent+2])
        mid_resp = mid_resp.reshape(np.shape(mid_resp)[0], np.shape(mid_resp)[-1]**2)
        nc = norm_cov(mid_resp.T)
        nc_array[img_ind, layer_ind] = nc
        layer_ind += 1

#%%
plt.imshow(nc_array)
##%%
#plt.subplot(411)
#plt.imshow(net.params['conv1'][0].data[1,0]);plt.colorbar()
#plt.title('mean:' + str(np.mean(net.params['conv1'][0].data[1,0]))
#            + 'sd' +str(np.std(net.params['conv1'][0].data[1,0])))
#plt.subplot(412)
#plt.imshow(net.params['conv1'][0].data[0,0]);plt.colorbar()
#plt.subplot(413)
#plt.imshow(net.params['fc8'][0].data.reshape(2,14,14)[0]);plt.colorbar()
#plt.subplot(414)
#plt.imshow(net.params['fc8'][0].data.reshape(2,14,14)[1]);plt.colorbar()
#
#print(net.params['conv1'][0].data.max())
#
##%%
#mag = ((10*10*3*2) + (14*14*2))**-1
#edge = np.ones((3, 10, 10))*mag
#edge[:, :3] = -mag
#edge[:, 7:] = -mag
#
#blob = edge.copy()
#blob[:] = mag
##%%
#net.params['conv1'][0].data[0] = edge
#net.params['conv1'][0].data[1] = blob
#net.save(caffe_root + '/models/apc_net/apcnet_train_iter_1.caffemodel')