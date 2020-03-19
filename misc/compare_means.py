# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:23:26 2016

@author: dean
"""
import numpy as np
from collections import OrderedDict as ordDict

import os
import sys
import warnings

#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')

sys.path.append('/home/dean/caffe/python')

import caffe
meanweb = '/home/dean/caffe/data/imagenet_mean.binaryproto'
meanmine='/home/dean/caffe/data/ilsvrc12/imagenet_mean.binaryproto' 
meanmine='/home/dean/caffe/data/ilsvrc12/imagenet_mean_me.binaryproto' 


blob = caffe.proto.caffe_pb2.BlobProto()
data = open( meanweb , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out1 = arr[0]

data = open( meanmine , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out2 = arr[0]
print(out1)
print(out2)
print('diff')
print(out1-out2)