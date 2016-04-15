# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:38:46 2016

@author: dean
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#make the working directory two above this one
top_dir = os.getcwd().split('net_code')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir +'common')

caffe_root = '/home/dean/caffe/'
sys.path.insert(0, caffe_root + 'python')

import d_misc as dm
import xarray as xr
import caffe_net_response as cf
import pandas as pd

def uniquer(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

da = xr.open_dataset(top_dir + 'nets/alex_net_nat_image_dist2000.nc' )['r']
layers = dict(da.groupby('layer_label', ))
'''
for ind, key in enumerate(uniquer(da.coords['layer_label'].values)):
    print(key)
    layer = layers[key]
    vals = layer.values.flatten()
    vals = vals[vals>0]
    hist, bins = np.histogram(vals, bins=1000)

    plt.subplot(len(layers)/2, 2, ind+1)
    plt.plot(bins[0:-1], hist[0:])
    plt.ylabel(key)
    plt.yticks([0, max(hist[0:])])
#    plt.gca().set_xticklabels(['0', .25, .5, .75, 1])

plt.tight_layout()
'''
resp = da.values
resp = [unit[~(0==unit)] for unit in resp.T ]

quart = np.array([ np.percentile(unit, [25,75], axis=0) for unit in resp if unit.shape[0]>0])
plt.plot(quart)