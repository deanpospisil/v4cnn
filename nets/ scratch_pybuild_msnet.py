#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:15:18 2017

@author: dean
"""

#%%
import os
import sys
import re
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0] 
sys.path.append(top_dir+ 'v4cnn')
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')
sys.path.append( top_dir + 'nets/')

import caffe_net_response as cf
import d_misc as dm
import xarray as xr
import apc_model_fit as ac
import d_curve as dc
import d_img_process as imp
import scipy.io as l
import caffe
from caffe import layers as L
from caffe import params as P

def layer_txt(net_params):
    #hand a list of net params the first column is the type of param and the second the value
    # a list within a list in the values section is a subparam whose
    nettxt = []
    nettxt.append('layer {')
    for key, val in net_params:
    
    nettxt.append('}')
    
    return l

n = caffe.NetSpec()

n.data, n.label = L.Data(batch_size=256, transform_param=dict(mirror=True, crop_size=227),
                                ntop=2)

#the first layer is spatial
n.conv1 = L.Convolution(n.data, ntop=1, convolution_param=dict(num_output=256,
                                                              kernel_size=4,
                                                              stride=2))

#now we start the actual units
n.conv2 = L.Convolution(n.conv1, ntop=1, convolution_param=dict(num_output=128,
                                                              kernel_size=1,
                                                              stride=1))


#spatial conv
n.conv = L.Convolution(n.conv1, ntop=1, convolution_param=dict(num_output=128,
                                                              kernel_size=3,
                                                              stride=1,
                                                              group=32))



print(n.to_proto())
#%%
def example_network(batch_size):
    n = caffe.NetSpec()

    n.loss, n.love = L.Data(shape=[dict(dim=[1]),
                                         dict(dim=[1])],
                                  transform_param=dict(scale=1.0/255.0),
                                  ntop=2)

    n.accuracy = L.Python(n.loss, n.love,
                          python_param=dict(
                                          module='python_accuracy',
                                          layer='PythonAccuracy',
                                          param_str='{ "param_name": param_value }'),
                          ntop=1,)

    return n.to_proto()

#print(example_network(5))
#just do it yourself
#%%
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

net = caffe_pb2.NetParameter()

fn = '/home/dean/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
with open(fn) as f:
    s = f.read()
txtf.Merge(s, net)

net.name = 'my new net'
layerNames = [l.name for l in net.layer]
idx = layerNames.index('fc6')
l = net.layer[idx]
l.param[0].lr_mult = 1.3

outFn = '/tmp/newNet.prototxt'
print 'writing', outFn
with open(outFn, 'w') as f:
    f.write(str(net))
    
#you'll need to append layers using a dummy output input and then chage them
#after the fact./
#%%
import caffe
from caffe import layers as L #pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P #pseudo module using __getattr__ magic to generate protobuf messages
 
 
def setLayers(leveldb, batch_size, deploy=False):
    #it is tricky to produce the deploy prototxt file, as the data input is not 
    #from a layer, so we have to creat a workaround
    #producing training and testing prototxt files is pretty straight forward
    n = caffe.NetSpec()
    if deploy==False:
        n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LEVELDB, source=leveldb,\
        ntop=2)
    # produce data definition for deploy net
    else:
        input="data"
        dim1=1
        dim2=1
        dim3=64
        dim4=32
    #make an empty "data" layer so the next layer accepting input will be able to take the correct blob name "data",
    #we will later have to remove this layer from the serialization string, since this is just a placeholder
    n.data=L.Layer()
     
    n.conv2 = L.Convolution(n.data, kernel_h=6, kernel_w=1, num_output=8, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv2, kernel_h=3, kernel_w=1, stride_h=2, stride_w=1, pool=P.Pooling.MAX)
     
    n.drop2=L.Dropout(n.pool1,dropout_ratio=0.1)
    n.ip1=L.InnerProduct(n.drop2, num_output=196, weight_filler=dict(type='xavier'))
     
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip4 = L.InnerProduct(n.relu1, num_output=12, weight_filler=dict(type='xavier'))
     
    #n.loss layer is only in training and testing nets, but not in deploy net.
    if deploy==False:
        n.loss = L.SoftmaxWithLoss(n.ip4, n.label)
        return str(n.to_proto())
    #for generating the deploy net
    else:
    #generate the input information header string
        deploy_str='input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"'+input+'"', dim1, dim2, dim3, dim4)
    #assemble the input header with the net layers string.  remove the first placeholder layer from the net string.
    return deploy_str+'\n'+'layer {'+'layer {'.join(str(n.to_proto()).split('layer {')[2:])
     
    #write the net prototxt files out
    with open('trainNet.prototxt', 'w') as f:
        print 'wrting train'
        f.write(setLayers('databases/train_subj1_leveldb', 100))
     
    with open('testNet.prototxt', 'w') as f:
        print 'wrting test'
        f.write(setLayers('databases/test_subj1_leveldb', 100))
     
    with open('deploy.prototxt', 'w') as f:
        f.write(str(setLayers('', 0, deploy=True)))

#%%
def mnist_network(lmdb_path, batch_size):
    """
    Convolutional network for MNIST classification.
    
    :param lmdb_path: path to LMDB to use (train or test LMDB)
    :type lmdb_path: string
    :param batch_size: batch size to use
    :type batch_size: int
    :return: the network definition as string to write to the prototxt file
    :rtype: string
    """
        
    net = caffe.NetSpec()
        
    net.data, net.labels = caffe.layers.Data(batch_size = batch_size, 
                                             backend = caffe.params.Data.LMDB, 
                                             source = lmdb_path, 
                                             transform_param = dict(scale = 1./255), 
                                             ntop = 2)
    net.augmented_data = caffe.layers.Python(net.data, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationMultiplicativeGaussianNoiseLayer'))
    net.augmented_labels = caffe.layers.Python(net.labels, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationDoubleLabelsLayer'))
    
    net.conv1 = caffe.layers.Convolution(net.augmented_data, kernel_size = 5, num_output = 20, 
                                         weight_filler = dict(type = 'xavier'))
    net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size = 2, stride = 2, 
                                     pool = caffe.params.Pooling.MAX)
    net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size = 5, num_output = 50, 
                                         weight_filler = dict(type = 'xavier'))
    net.pool2 = caffe.layers.Pooling(net.conv2, kernel_size = 2, stride = 2, 
                                     pool = caffe.params.Pooling.MAX)
    net.fc1 =   caffe.layers.InnerProduct(net.pool2, num_output = 500, 
                                          weight_filler = dict(type = 'xavier'))
    net.relu1 = caffe.layers.ReLU(net.fc1, in_place = True)
    net.score = caffe.layers.InnerProduct(net.relu1, num_output = 10, 
                                          weight_filler = dict(type = 'xavier'))
    net.loss =  caffe.layers.SoftmaxWithLoss(net.score, net.augmented_labels)
        
    return net.to_proto()