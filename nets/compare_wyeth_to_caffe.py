#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:01:02 2018

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:43:31 2016

@author: dean
"""
import os
import sys
import numpy as np

#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'common')

caffe_root = '/home/dean/caffe/'
sys.path.insert(0, caffe_root + 'python')

import d_misc as dm
import xarray as xr
import caffe_net_response as cf
import caffe
#%%
img_dir= '/loc6tb/data/images/ILSVRC2012_img_val/'
img_names = [name for name in os.listdir(img_dir) if 'JPEG' in name]
#takes stim_trans_cart_dict, pulls from img_stack and transform accordingly,
    #gets nets responses.

nimgs_per_pass = 260
n_imgs = 100

if n_imgs>len(img_names):
    n_imgs = len(img_names)
stack_indices, remainder = dm.sectStrideInds(nimgs_per_pass, n_imgs)

import caffe
model_def = caffe_root + 'models/v4_model/bvlc_reference_caffenet/deploy.prototxt'
model_def = caffe_root + 'models/v4_model/caffe_model.prototxt'
model_weights = caffe_root + 'models/v4_model/bvlc_reference_caffenet.caffemodel'

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

#%%
all_net_resp = []
layer_names = [k for k in net.blobs.keys()]
for stack_ind in stack_indices:
    print(stack_ind)
    images = [caffe.io.load_image(img_dir + name) for name in img_names[stack_ind[0]:stack_ind[1]]]
    transformed_images = []
    for i, image in enumerate(images):
        transformed_images.append(transformer.preprocess('data', image))
    stack = np.array(transformed_images)
    #shape the data layer, (first layer) to the input
    net.blobs[ layer_names[0]].reshape(*tuple([stack.shape[0],]) + net.blobs['data'].data.shape[1:])
    net.blobs[layer_names[0]].data[...]= stack
    net.forward()

    all_layer_resp = []
    layer_names_sans_data = layer_names[1:]
    for layer_name in layer_names_sans_data:

        layer_resp = net.blobs[layer_name].data

        if len(layer_resp.shape)>2:#ignore convolutional repetitions, just pulling center.
            mid = [ round(m/2) for m in np.shape(net.blobs[layer_name].data)[2:]   ]
            layer_resp = layer_resp[ :, :, int(mid[0]), int(mid[1])]

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
da['img_name'] = ('img', img_names[:n_imgs])

#%%
ds = xr.Dataset({'r':da})
ds.to_netcdf(save_dir)

#%%
save_dir = '/loc6tb/data/responses/' + 'caffe_ilsvrc2012_val_resp.nc'

da = xr.open_dataset(save_dir)['r']
#%%
from scipy import misc
example_units = [('conv2', 113), ('conv2', 108), ('conv2', 126), ('conv3', 156), ('conv3', 20),
 ('conv5', 161), ('conv5', 144), ('conv3', 334), ('conv4', 203), ('fc6', 3030), 
 ('fc7', 3192), ('fc7', 3591), ('fc7', 3639), ('fc8', 271), ('fc8', 433), ('fc8', 722)]

example_units = [('conv3', 64),('conv3', 261), 
                 ('conv3', 278), ('conv3', 340)]

example_units = [('conv4', 0), ('conv4', 1),('conv3', 97), 
                 ('conv3', 133), ('conv3', 255)]

fn = 'bvlc_reference_caffenetpix_width[32.0]_x_(114.0, 114.0, 1)_y_(32, 196, 83)_amp_NonePC370_analysis.p'
fn = 'bvlc_reference_caffenetpix_width[32.0]_x_(32, 196, 83)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p'
fn = '/loc6tb/data/an_results/' + fn

import pickle as pk
try:
    an=pk.load(open(fn,'rb'), 
               encoding='latin1')
except:
    an=pk.load(open(fn,'rb'))
    
cn_p = an[1]

cn_p['ti_in_rf'].loc['conv3']
#%%
img_dir= '/loc6tb/data/images/ILSVRC2012_img_val/'

layer_label = da.coords['layer_label'].values
layer_unit = da.coords['layer_unit'].values

for i, unit in enumerate(example_units):
    ind = (layer_unit == unit[1])*(layer_label==unit[0])
    unit_resp = da[:, np.argmax(ind)]
    resp_sort_ind = np.argsort(unit_resp,0).values[::-1]
    plt.figure(figsize=(2,10))        
    for j, ind in enumerate(resp_sort_ind[100:105]):
        plt.subplot(5,1,j+1)    
        if j==0:
            plt.title(unit)
        fn = str(np.squeeze(unit_resp[ind].coords['img_name']).values)
        img = misc.imread(img_dir + fn)
        plt.imshow(img)
        plt.xticks([]);plt.yticks([])
    

#        plt.savefig(top_dir + '/analysis/figures/images/v4cnn_cur/'+
#            'mid_resp'+ str(unit) +'.pdf', bbox_inches='tight',
#            dpi=500)