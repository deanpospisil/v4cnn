# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:19:47 2016

@author: dean
"""
import numpy as np
import os
import sys

#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')
sys.path.append(top_dir +'/nets')

import d_img_process as imp
import d_misc as dm
import d_curve as dc
import scipy.io as  l
import matplotlib.pyplot as plt


import d_curve as dc

import caffe_net_response as cf
import d_img_process as imp



img_n_pix = 227
max_pix_width = [32.,]

s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
s = np.load(top_dir + 'img_gen/dp_ang_pos_verts.npy')
s = np.load(top_dir + 'img_gen/dp_ang_pos_verts_shift.npy')

base_stack = imp.center_boundary(s)


#%%
def stim_trans_generator(shapes=None, shapes2=None, offsetsx=None, blur=None, scale=None,
                         x=None, y=None, amp=None, rotation=None):
#takes descrptions of ranges for different transformations (start, stop, npoints)
#produces a cartesian dictionary of those.
    stim_trans_dict = ordDict()
    if shapes is not None:
        #stim_trans_dict['shapes'] = np.array(shapes, dtype=int)
        stim_trans_dict['shapes'] = np.array(shapes)
    
    if shapes2 is not None:
        #stim_trans_dict['shapes'] = np.array(shapes, dtype=int)
        stim_trans_dict['shapes2'] = np.array(shapes2)

    if blur is not None :
        if isinstance(blur,tuple):
            stim_trans_dict['blur'] = np.linspace(*blur)
        else:
            stim_trans_dict['blur'] = blur
    if scale is not None:
        if isinstance(scale, tuple):
            stim_trans_dict[ 'scale' ] = np.linspace(*scale)
        else:
            stim_trans_dict[ 'scale' ] = scale
    if  x is not None :
        if type(x) is tuple:
            stim_trans_dict['x'] = np.linspace(*x)
        else:
            stim_trans_dict['x'] = np.array(x)
            
    if y is not None :
        if type(y) is tuple:
            stim_trans_dict['y'] = np.linspace(*y)
        else:
            stim_trans_dict['y'] = np.array(y)
    
    if offsetsx is not None:
        if type(offsetsx) is tuple:
            stim_trans_dict['offsetsx'] = np.linspace(*offsetsx)
        else:
            stim_trans_dict['offsetsx'] = np.array(offsetsx)
            
    if  rotation is not None :
        if type(rotation) is tuple:
            stim_trans_dict['rotation'] = np.linspace(*y)
        else:
            stim_trans_dict['rotation'] = np.array(rotation)
            
    if not amp is None:
        stim_trans_dict['amp'] = np.linspace(*amp)
        # get all dimensions, into a dict
    stim_trans_cart_dict = dm.cartesian_prod_dicts_lists(stim_trans_dict)
    #this is a hackish fix to cart from sklearn switching dtypes according to 1st element.
    stim_trans_cart_dict['shapes'] = stim_trans_cart_dict['shapes'].astype(int)
    return stim_trans_cart_dict, stim_trans_dict

from itertools import product


sign = [1, -1]
scale = [16,]
img_n_pix = 227





from collections import OrderedDict as ordDict
stim_trans_cart_dict = ordDict()


#stim_trans_dict['shapes'] = [np.array(e1, e2) for e1, e2 in zip(stim_cart, stim_cart2)]
center_image = round(img_n_pix/2)
x = (center_image, center_image, 1)
y = (center_image, center_image, 1)
offsetsx = np.array(list(max_pix_width*np.array([0.5, 1, 2])))
shape_ids = np.arange(-1, 362, 6)

scale = max_pix_width/dc.biggest_x_y_diff(base_stack)
stim_trans_cart_dict, stim_trans_dict = stim_trans_generator(shapes=shape_ids,
                     shapes2=shape_ids,
                     scale=scale,
                     x=x,
                     offsetsx=offsetsx,
                     y=y)

#%%
#y = (center_image, center_image, 11)
#amp = (255, 255, 1)
#amp = None
#stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
#                                                                scale=scale,
#                                                                x=x,
#                                                                y=y,
#                                                                amp=amp,
#                                                                rotation=(0, 360,10))

n_rot = 5
rotation = np.linspace(*(0, np.deg2rad(360-360./n_rot), n_rot))
rotation=None
scale = max_pix_width/dc.biggest_x_y_diff(base_stack)
shape_ids = np.arange(0., 18.)
center_image = round(img_n_pix/2)
x = (center_image, center_image + 48, 1)
y = (center_image, center_image, 1)
stim_trans_cart_dict, stim_trans_dict = stim_trans_generator(shapes=shape_ids,
                                                         
                                        scale=scale,
                                        x=x,
                                        y=y,)
                                        #rotation=rotation)
stim_trans_cart_dict, stim_trans_dict = stim_trans_generator(shapes=shape_ids,
                     shapes2=shape_ids,
                     scale=scale,
                     x=x,
                     offsetsx=offsetsx,
                     y=y)
#%%
figure_folder = top_dir + 'analysis/figures/images/'
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict,
                                                        base_stack, npixels=227))
#plot smallest and largest shape
no_blank_image = trans_img_stack[:]

for ind in range(len(no_blank_image))[:5]:
    plt.figure(figsize=(6,12))
    plt.imshow(no_blank_image[ind], cmap=plt.cm.Greys_r)
    #plt.title(str(stim_trans_cart_dict['shapes'][ind]) + ' '+
    #          str(np.round(np.rad2deg(stim_trans_cart_dict['rotation'][ind]))))
    plt.xticks([]);plt.yticks([])




