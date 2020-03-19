<<<<<<< HEAD
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:31:59 2018

@author: dean
"""

import numpy as np
import matplotlib.pyplot as plt
import os
im_folder = '/loc6tb/data/images/ILSVRC2012_img_val_windowed_softer/'
all_fn = os.listdir(im_folder)
all_fnp = [int(fn.split('.')[0]) for fn in all_fn if '.png' in fn]
#%%
biggest = np.max(all_fnp)

blank_ind = biggest + 1
#
##%%
#
#im = plt.imread(im_folder+'0.png' )
#
#im[...] = 0
#
#plt.imsave(im_folder+str(blank_ind)+'.png', im)
#plt.imsave(im_folder+str(blank_ind)+'.bmp', im)
#
##%%
#plt.imshow(plt.imread(im_folder+str(blank_ind)+'.png'))
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:14:55 2016

@author: deanpospisil
"""

import pickle
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

import itertools
flatten_iter = itertools.chain.from_iterable
def factors(n):
    return set(flatten_iter((i, n//i) 
                for i in range(1, int(n**0.5)+1) if n % i == 0))

def vis_square(ax, data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    ax.imshow(data, interpolation='nearest'); ax.axis('off')

def chrom_index(im):
    im_power = np.sum(im**2)
    im_var = np.sum((im - np.mean(im, -1, keepdims=True))**2)
    return im_var/im_power        
if 'a' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:    
        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)
        wts = np.transpose(a[0][1], (0,2,3,1))

val_map = []
for a_filter in wts:
    a_filter_unwrap = a_filter.reshape(np.product(a_filter.shape[:-1]),3)
    u,s, v = np.linalg.svd(a_filter_unwrap, full_matrices=False)
    val_map.append(np.dot(a_filter_unwrap, v[0,:]).reshape(np.shape(a_filter)[:-1]))
val_map = np.array(val_map)
ft_val_map = np.abs(np.fft.fft2(val_map))

ft_val_map = np.array([np.fft.fftshift(a_map) for a_map in ft_val_map])
index = np.linspace(-1,1,11)
x, y = np.meshgrid(index,index)
freq=(x**2 + y**2)**0.5
#freq_sort =np.argsort(freq.ravel())

top_freq = np.array([freq.ravel()[a_map.ravel().argmax()] for a_map in ft_val_map])

c = np.array([chrom_index(im) for im in wts])
ax = plt.subplot(121)
ax.scatter(c, top_freq)
ax.set_xlabel('Chromaticity')
ax.set_ylabel('Peak Frequency')
ax = plt.subplot(122)
sort_freq = np.array([wts[freq_ind] for freq_ind in np.argsort(top_freq)])
vis_square(ax, sort_fre)
>>>>>>> 4b9ff218dfb66876a9eca6a5220ceffaa6d987fb
