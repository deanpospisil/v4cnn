# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:45:56 2016

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:35:06 2016

@author: dean
"""
import os
import sys
import warnings
import numpy as np
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')
ann_fn = 'bvlc_reference_caffenet'
sys.path.append('/home/dean/caffe/python')
import caffe
caffe.set_mode_gpu()
net = caffe.Net(ann_dir + 'deploy_fixing_relu_saved.prototxt', ann_dir + ann_fn + '.caffemodel', caffe.TEST)
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
print([net.params[name][0].data.shape for name in layer_names])

#for i, name in enumerate(layer_names):
#    wts = net.params[name][0].data
#    wts_f = wts.reshape((wts.shape[0], np.product(wts.shape[1:])))
#    net.params[name][0].data[...] = np.random.permutation(wts_f.T).T.reshape(wts.shape)
#    net.save( ann_dir+ 'bvlc_caffenet_reference_shuffle_layer_'+ str(i) +'.caffemodel')
#    net.params[name][0].data[...] = wts



