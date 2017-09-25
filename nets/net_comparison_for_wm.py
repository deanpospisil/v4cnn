#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:36:56 2017

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:33:15 2016

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:56:23 2016

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

img_names = os.listdir(caffe_root + 'examples/images/')
for img_ind, an_image in enumerate(img_names):
    image = caffe.io.load_image(caffe_root + 'examples/images/' + an_image)
    ti = transformer.preprocess('data', image)
    f = open('/home/dean/Desktop/v4cnn/images/img_for_wyeth_comparison/'+ str(img_ind) +'.txt', 'w')
    for dim in np.shape(ti):
        f.write(str(dim) + ' ')
    f.write('\n')
    for v in ti.flatten():
        f.write(str('{0:.16f}'.format(v)) + ' ')
    f.close()
    net.blobs['data'].data[...] = ti
    output = net.forward()
    
    f = open('/home/dean/Desktop/v4cnn/data/responses/alex_resp_for_wyeth/an_resp_'+ str(img_ind) +'.txt', 'w')
    layers = list(net.blobs.iteritems())
    for layer_name, blob in layers:
        print layer_name + '\t' + str(blob.data.shape)
        if not ('fc' in layer_name or 'prob' in layer_name or '6' in layer_name
                or '7' in layer_name or '8' in layer_name):
            center = int(np.ceil(blob.data.shape[-1]/2.)-1)
            print(center)
            layer = blob.data[0, :, center, center]
            
            #for dim in np.shape(layer):
            #    f.write(str(dim) + ' ')               
            #f.write('\n')
            for ind, w in enumerate(layer.flatten()):
                f.write(layer_name + '_' + str(center) + '_' + str(center) + '_' + str(ind) + ' ')
                f.write(str('{0:.16f}'.format(w)))
                f.write('\n')
        else:
            layer = blob.data[0, :]
            for w in layer.flatten():
                f.write(layer_name + ' ')
                f.write(str('{0:.16f}'.format(w)))
                f.write('\n')
    
    f.close()   