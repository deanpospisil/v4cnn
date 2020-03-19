# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:11:57 2016

@author: dean
"""


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
sys.path.append('/home/dean/miniconda/lib/python2.7/site-packages')
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

lmdb_env = lmdb.open('/home/dean/caffe/examples/imagenet/ilsvrc12_train_lmdb/')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()
labels=[]
keys =[]
x=0
for key, value in lmdb_cursor:
    
    datum.ParseFromString(value)
    label = datum.label
    labels.append(label)
    keys.append(key)

labels=np.array(labels)
np.save('labels', np.array(labels))

import pickle as pk
pk.dump( keys, open( 'key.p', "wb" ) )
#for key, value in lmdb_cursor:
#    datum.ParseFromString(value)
#    x+=1
#    print(x)
#    label = datum.label
#    labels.append(key)
#    print(label)
#    print(key)
#    data = caffe.io.datum_to_array(datum)
#    image = np.transpose(data, (1,2,0))
    #print(image.shape)
#    if x > 10:
#            break