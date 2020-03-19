#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:29:49 2017

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

import caffe_net_response as cf
import d_img_process as imp

import matplotlib.pyplot as plt


img_n_pix = 227
max_pix_width = [ 100.,]

mat = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')
s = np.array(mat['shapes'][0])

base_stack = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(base_stack)
shape_ids = range(-1, 370)
center_image = round(img_n_pix/2.)
x = (center_image, center_image, 1)
y = (center_image, center_image, 1)
offsetsx = np.array(list(max_pix_width*np.array([0.5, 1, 2])))
shape_ids = np.arange(-1, 370, 1)

scale = max_pix_width/dc.biggest_x_y_diff(base_stack)
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                     scale=scale,
                     x=x,
                     y=y)




figure_folder = top_dir + 'analysis/figures/images/'
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict,
                                                        base_stack, npixels=227))[1:]
trans_img_stack = np.tile(trans_img_stack, [3,1,1,1])

#%%
sys.path.append('/home/dean/caffe/python')
caffe_root = '/home/dean/caffe/'
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ann_fn = 'bvlc_reference_caffenet'

import caffe 
caffe.set_mode_gpu()

net = caffe.Net(ann_dir + 'deploy_fixing_relu_saved.prototxt', 
                ann_dir + ann_fn + '.caffemodel', caffe.TEST)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 
                          227)  # image size is 227x227
         
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', np.array([0,0,0]))            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


img_dir ='/loc6tb/data/images/apc_imgs_for_wyeth/'
for img_ind in range(370):
    print(img_ind)
    # image = caffe.io.load_image(example_image_dir + an_image)
    image = np.squeeze(trans_img_stack[:, img_ind, ...])
    ti = transformer.preprocess('data', image)
#    np.save(img_dir + str(img_ind), ti)

    f = open(img_dir + str(img_ind) + '.txt', 'w')
    for dim in np.shape(ti):
        f.write(str(dim) + ' ')
    f.write('\n')
    for v in ti.flatten():
        f.write(str('{0:.16f}'.format(v)) + ' ')
    f.close()    