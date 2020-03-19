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

net = caffe.Net(ann_dir + 'deploy.prototxt', ann_dir + ann_fn + '.caffemodel', caffe.TEST)
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
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
ti = transformer.preprocess('data', image)
f = open('/home/dean/cat.txt', 'w')
for dim in np.shape(ti):
    f.write(str(dim) + ' ')
f.write('\n')
for v in ti.flatten():
    f.write(str(v) + ' ')
f.close()
#%%
net.blobs['data'].data[...] = ti
output = net.forward()
f = open('/home/dean/an_cnv_resp.txt', 'w')
layers = list(net.blobs.iteritems())
for layer_name, blob in layers:
    print layer_name + '\t' + str(blob.data.shape)
    if not ('fc' in layer_name or 'prob' in layer_name):
        print(blob.data[0,0,:5,:5] )
        
        layer = blob.data[0,0,:5,:5]
        
        #for dim in np.shape(layer):
        #    f.write(str(dim) + ' ')               
        #f.write('\n')
        for w in layer.flatten():
            f.write(layer_name + ' ')
            f.write(str(w))
            f.write('\n')
    else:
        print(blob.data[0,:20] )
        layer = blob.data[0,:20]
        f.write('LAYER ' + layer_name + ' response ')
        for dim in np.shape(layer):
            f.write(str(dim) + ' ')               
        f.write('\n')
        for w in layer.flatten():
            f.write(str(w) + ' ')
        f.write('\n')
        

f.close()        


'''
#kernels ()
print('kernel')
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
    if not ('fc' in layer_name or 'prob' in layer_name):
        mid=np.ceil(param[0].data.shape[-1]/2.)
        print(mid)
        print(param[0].data[0,0,0,0])
        print('weight')
        print(param[1].data[0])
        print('bias')
    else:
        print(param[0].data[0,0])
        print('weight')
        print(param[1].data[0])
        print('bias')
    
#outputs
layer_names = []
print('response')
for layer_name, blob in net.blobs.iteritems():
    layer_names.append(layer_name)
    print layer_name + '\t' + str(blob.data.shape)
    if not ('fc' in layer_name or 'prob' in layer_name):
        print(blob.data[0,0,0,0] )
    else:
        print(blob.data[0,0])
ln=['data', 'conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4',
 'conv5', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']    
for k in [0,1,2]:
    a = net.params[ln[1]][0].data[0,k,:,:]
    print(np.sum(a))
#b = net.params[ln[1]][1].data[0]
#print(a)
#print(np.sum(a))
#print(np.sum(a)*255)+ b
#print(b)
'''