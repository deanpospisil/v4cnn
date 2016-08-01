# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:33:44 2016

@author: dean
"""

#stimfo
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common')
sys.path.append( top_dir + 'nets')
import xarray as xr
import d_img_process as imp
import pandas as pd
import caffe_net_response as cf



#choose a library of images
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]
img_dir = top_dir+'/images/baseimgs/'+ base_image_nm + '/'
base_stack, stack_desc = imp.load_npy_img_dirs_into_stack(img_dir)
scale = 1.

stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(
                                                 shapes=range(-1, 370),
                                                 blur=None,
                                                 scale=(scale, scale, 1),
                                                 x=None,
                                                 y=None,
                                                 rotation=None)
trans_img_stack = imp.imgStackTransform(stim_trans_cart_dict, base_stack)
base_stack = trans_img_stack



im_ids = [int(re.findall('\d+[.npy]', fn)[0][:-1]) for fn in stack_desc['img_paths']]
im_size = base_stack.shape[-1]

#area
im_area = np.array([np.sum(img>0)/np.prod(np.shape(img)).astype(float)
            for img in base_stack if np.sum(img>0)>0])
#upmost row, downmost row, leftmost col, rightmost col

t = np.array([ ind for ind, img in enumerate(base_stack) if np.sum(img>0)>0])
im_edge = np.array([[np.nonzero(img.sum(1))[0][0], np.nonzero(img.sum(1))[0][-1],
            np.nonzero(img.sum(0))[0][0], np.nonzero(img.sum(0))[0][-1]]
            for img in base_stack if np.sum(img>0)>0])
im_edge = np.array(im_edge)
im_power = [np.sum(img**2) for img in base_stack if np.sum(img>0)>0]

var_names = [ 'area','power', 'up', 'down', 'left', 'right']
img_vars = np.column_stack([im_area, im_power,
                            im_edge[:,0], im_edge[:,1], im_edge[:,2], im_edge[:,3]])
im_info = pd.DataFrame(img_vars, columns=var_names, index=range(370))

smallest_width = (im_info['right'] - im_info['left']).min()
widest_width = (im_info['right'] - im_info['left']).max()
w
left_most_ind = im_info.left.argmin()
left_shift_lim = -im_info['left'][left_most_ind]


right_most_ind = im_info.right.argmax()
right_shift_lim = im_size - im_info['right'][right_most_ind]

steps = np.arange(left_shift_lim, right_shift_lim, smallest_width/2.)
import matplotlib.cm as cm

for ind, img in enumerate(base_stack[:25]):
    plt.subplot(5,5, ind+1)
    plt.imshow(img.squeeze(), interpolation='nearest', cmap = cm.Greys_r)