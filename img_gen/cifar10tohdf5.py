#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:57:56 2017

@author: dean
"""



import h5py
import os
import numpy as np
import cPickle

def unpickle(a_file):

    with open(a_file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
img_dir = '/loc6tb/data/images/cifar-10/cifar-10-batches-py/'
img_batch = 'data_batch_1'
img_batch = 'test_batch'


blah = unpickle(img_dir + img_batch)
data = blah['data']
data = data.reshape(np.shape(data)[0], 3, 32, 32)
label = np.array(blah['labels'])

DIR = img_dir
h5_fn = os.path.join(DIR, 'cifar10_' + img_batch + '.h5')

with h5py.File(h5_fn, 'w') as f:
   f['data'] = data
   f['label'] = label
   
   f.close()

text_fn = os.path.join(DIR, 'cifar10_dir_' + img_batch + '.txt')
f = open(text_fn,'w')
f.write(h5_fn)
f.close()
    