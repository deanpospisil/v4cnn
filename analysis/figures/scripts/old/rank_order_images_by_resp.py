# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:24:18 2016

@author: dean
"""
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import os, sys
import matplotlib.patches as mpatches
import matplotlib.cm as cm

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=0, padval=0):
    plt.figure(figsize = (10,7.8))
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data, interpolation='None', cmap = cm.Greys_r)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel('Normalized Firing Rate', rotation= 270, labelpad=15, fontsize = 15,)
    cbar.ax.yaxis.set_ticks([0,.25,.5,.75, 1])
    cbar.ax.set_yticklabels(['0', .25, .5, .75, 1])
    cbar.solids.set_rasterized(True)
    plt.tight_layout()

s = targets
s = s*0.8 + .2
s.shape= (370,1,1)
imgStack = imgStack*(s)
s.shape= (370,)

sortShapesInds = np.argsort(s)[-1::-1]
sortResp = s[sortShapesInds]

sortStack = imgStack[sortShapesInds,:,:]

vis_square(sortStack[0:49])
plt.xticks([])
plt.yticks([])
plt.title('Shape Response: Cell ' + str(cell) )