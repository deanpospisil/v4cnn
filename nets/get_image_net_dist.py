# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:43:31 2016

@author: dean
"""
import os
import sys
import numpy as np

#make the working directory two above this one
top_dir = os.getcwd().split('net_code')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir +'common')

caffe_root = '/home/dean/caffe/'
sys.path.insert(0, caffe_root + 'python')

import d_misc as dm
import xarray as xr
import caffe_net_response as cf

img_dir= top_dir +'data/image_net/'
img_names = [name for name in os.listdir(img_dir) if 'JPEG' in name]
#takes stim_trans_cart_dict, pulls from img_stack and transform accordingly,
    #gets nets responses.

nimgs_per_pass = 260
n_imgs = 2000

if n_imgs>len(img_names):
    n_imgs = len(img_names)
rand_img_inds = np.random.choice(len(img_names), n_imgs)
stack_indices, remainder = dm.sectStrideInds(nimgs_per_pass, n_imgs)

import caffe
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
images = [caffe.io.load_image(img_dir + name) for name in [img_names[ind] for ind in rand_img_inds]]
transformed_images = [transformer.preprocess('data', image) for image in images]



#now divide the dict up into sects.
#order doesn't matter using normal dict, imgStackTransform has correct order
all_net_resp = []
layer_names = [k for k in net.blobs.keys()]
for stack_ind in stack_indices:
    stack = np.array(transformed_images[stack_ind[0]:stack_ind[1]])
    #shape the data layer, (first layer) to the input
    net.blobs[ layer_names[0]].reshape(*tuple([stack.shape[0],]) + net.blobs['data'].data.shape[1:])
    net.blobs[ layer_names[0]].data[...]= stack
    net.forward()

    all_layer_resp = []
    layer_names_sans_data = layer_names[1:]
    for layer_name in  layer_names_sans_data:

        layer_resp = net.blobs[layer_name].data

        if len(layer_resp.shape)>2:#ignore convolutional repetitions, just pulling center.
            mid = [ round(m/2) for m in np.shape(net.blobs[layer_name].data)[2:]   ]
            layer_resp = layer_resp[ :, :, mid[0], mid[1] ]

        all_layer_resp.append(layer_resp)
    response = np.hstack( all_layer_resp )
    all_net_resp.append(response)


#stack up all these responses
response = np.vstack(all_net_resp)
indices_for_net_unit_vec = cf.get_indices_for_net_unit_vec(net)
da = xr.DataArray(response, dims=['img', 'unit'])

# adding extra coordinates using indices_for_net_unit_vec
d = indices_for_net_unit_vec
da['layer'] = ('unit', d['layer_ind'])
da['layer_unit'] = ('unit', d['layer_unit_ind'])
layer_label = [d['layer_names'][int(layer_num)] for layer_num in d['layer_ind']]
da['layer_label'] = ('unit', layer_label)
ds = xr.Dataset({'r':da})
ds.to_netcdf(top_dir + 'data/an_results/alex_net_nat_image_dist2000.nc')

xr.load(top_dir + 'data/an_results/alex_net_nat_image_dist.nc' )

#randomly select n images in you image dir
#
#cf.net_imgstack_response(net, stack)