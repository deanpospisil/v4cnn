#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:15:40 2018

@author: dean
"""

import caffe as cf
import sys
import matplotlib.pyplot as plt
blob = cf.proto.caffe_pb2.BlobProto()
data = open('/home/dean/caffe.orig/data/imagenet_mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( cf.io.blobproto_to_array(blob) )
out = arr[0]
temp = np.copy(out)
out[0, ...] = temp[2, ...]
out[1, ...] = temp[1, ...]
out[2, ...] = temp[0, ...]


temp = out
out = out/out.max()
out[:,128,127] = np.array([1,1,1])
out[:,128,128] = np.array([0,0,0])


plt.imshow(out.T)
plt.savefig('/home/dean/Desktop/av_img.pdf')
#%%
plt.figure()
plt.hist(temp[0, ...].ravel(), histtype='step', color='b')
plt.hist(temp[1, ...].ravel(), histtype='step', color='g')
plt.hist(temp[2, ...].ravel(), histtype='step', color='r')

plt.savefig('/home/dean/Desktop/pix_dist.pdf')

