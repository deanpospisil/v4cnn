# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:02:39 2018

@author: deanpospisil
"""
import numpy as np
import matplotlib.pyplot as plt

def vis_square(ax, data, padsize=0, padval=0):
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    ax.set_xticks([]);ax.set_yticks([])
    return data


data = np.random.normal(size=(17, 10,10, 3))
fig = plt.figure()
ax = plt.gca()
data = vis_square(ax, data, padsize=1)
plt.imshow(data)
    

#I want to be able to make a rectangle

def vis_ims(data, r=None, c=None, padsize=0, padval=0):
    #expects data to be imgXrowXcolXRGB will provide tiling from bottom to top
    padding = ((0, r*c - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    data = data.reshape((r, c) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((r * data.shape[1], c * data.shape[3]) + data.shape[4:])
    return data.transpose([1,0,2])    
 
def show_ims(fig, data):
    ax = plt.gca()
    ax.set_xticks([]);ax.set_yticks([])
    ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False);ax.spines['left'].set_visible(False)
    plt.imshow(data)

data = np.random.normal(size=(16, 10, 10, 3))*np.linspace(0,1,16).reshape(16,1,1,1)
data -= data.min()
data /= data.max()
fig = plt.figure()

ax.set_xticks([]);ax.set_yticks([])
ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False);ax.spines['left'].set_visible(False)
data = vis_ims(data, r=3,c=7, padsize=1, padval=1)
plt.imshow(data)