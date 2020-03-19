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