# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:07:21 2016

@author: deanpospisil
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import signal
import pickle 
import os
top_dir = os.getcwd().split('v4cnn')[0]

goforit=True
if 'a' not in locals() or goforit:
    with open(top_dir + 'v4cnn/nets/netwts.p', 'rb') as f:    
        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)


wts = np.transpose(a[0][1], (0, 2, 3, 1))          
plt.subplot(111)
image = misc.imread(top_dir + '/v4cnn/images/wyeth.JPG')[:600:2,:600:2]
image = misc.imread(top_dir + '/v4cnn/images/jaywalking.JPG')[:600:2,:600:2]

ax = plt.subplot(111)
ax.imshow(image, interpolation='nearest')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(top_dir + '/v4cnn/analysis/figures/images/wyeth.jpg', bbox_inches='tight')

from skimage.measure import block_reduce


#image = np.swapaxes(image,0,1)
filt_ims = []
n_filts = 3
for i, a_filter in enumerate(wts[:n_filts]):
    filt_im = []
    for channel, filt_chan in zip(image.T, a_filter.T):
        filt_im.append(signal.convolve2d(channel, filt_chan, mode='same', boundary='symm'))
    filt_ims.append(np.sum(np.array(filt_im),0))

p_ims = []
for im, a_filter, i in zip(filt_ims, wts[:n_filts], range(n_filts)):
    im = im.T
    to_draw = a_filter.copy()
    to_draw -= to_draw.min()
    to_draw /= to_draw.max()
    ax.imshow(to_draw, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(top_dir + '/v4cnn/analysis/figures/images/'+str(i)+'_filt.jpg', bbox_inches='tight')

    ax.imshow(im, interpolation='nearest', cmap='Greys_r')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig(top_dir + '/v4cnn/analysis/figures/images/'+str(i)+'_conv.jpg', bbox_inches='tight')
    img_range = [np.min(im), np.max(im)]

    im =im[::3,::3]
    ax.imshow(im, interpolation='nearest', cmap='Greys_r')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(top_dir + '/v4cnn/analysis/figures/images/'+str(i)+'_conv_d_samp.jpg', bbox_inches='tight')
    
    im[im<50]=0
    ax.imshow(im, interpolation='nearest', cmap='Greys_r', vmin=img_range[0], 
              vmax=img_range[1])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(top_dir + '/v4cnn/analysis/figures/images/'+str(i)+'_conv_d_samp_relu.jpg', bbox_inches='tight')
    
    im = block_reduce(im, block_size=(4, 4), func=np.max)
    ax.imshow(im, interpolation='nearest', cmap='Greys_r', vmin=img_range[0], 
              vmax=img_range[1])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(top_dir + '/v4cnn/analysis/figures/images/'+str(i)+'_conv_d_samp_relu_pool.jpg', bbox_inches='tight')
    p_ims.append(im)
p_ims = np.array(p_ims)
alpha = 1
beta = 5
n = 3
normed_ims = p_ims/(1+(alpha/n)*np.sum(p_ims**2,0,keepdims=True))*beta

for i, im in enumerate(normed_ims):
    vmax = im.max()
    
    ax.imshow(im, interpolation='nearest', cmap='Greys_r', vmin=-vmax, 
              vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(top_dir + '/v4cnn/analysis/figures/images/'+str(i)+'_conv_d_samp_relu_pool_nrm.jpg', bbox_inches='tight')
    
    
    



