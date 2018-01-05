#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:53:39 2017

@author: dean
"""

import os
import sys
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
import numpy as np
sys.path.append('/home/dean/caffe/python')
import caffe
import matplotlib.pyplot as plt


    
caffe_root = '/home/dean/caffe/'
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ann_fn = 'bvlc_reference_caffenet'

caffe.set_mode_gpu()

net = caffe.Net(ann_dir + 'deploy_fixing_relu_saved.prototxt', ann_dir + ann_fn + '.caffemodel', caffe.TEST)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
example_image_dir = '/loc6tb/data/images/ILSVRC2012_img_val/'
img_names = os.listdir(example_image_dir)
img_names = [img_name for img_name in img_names if 'JPEG' in img_name]

for img_ind, an_image in enumerate(img_names):
    image = caffe.io.load_image(example_image_dir + an_image)
    ti = transformer.preprocess('data', image)
    f = open('/home/dean/Desktop/v4cnn/images/img_for_wyeth_comparison/'+ str(img_ind) +'.txt', 'w')
    for dim in np.shape(ti):
        f.write(str(dim) + ' ')
    f.write('\n')
    for v in ti.flatten():
        f.write(str('{0:.16f}'.format(v)) + ' ')
    f.close()
    
   
    f.close()   