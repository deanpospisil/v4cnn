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
n_rot = 10
rotation = np.linspace(*(0, np.deg2rad(360-360./n_rot), n_rot))
scale = max_pix_width/dc.biggest_x_y_diff(base_stack)
shape_ids = np.arange(0., 9.)
center_image = round(img_n_pix/2)
x = (center_image, center_image + 48, 1)
y = (center_image, center_image, 1)
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                                        scale=scale,
                                        x=x,
                                        y=y,
                                        rotation=rotation)

figure_folder = top_dir + 'analysis/figures/images/'
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict,
                                                        base_stack, npixels=227))
#plot smallest and largest shape
no_blank_image = trans_img_stack[:9]

for ind in range(len(no_blank_image)):
    plt.figure(figsize=(6,12))
    plt.imshow(no_blank_image[ind], cmap=plt.cm.Greys_r)
    plt.xticks([]);plt.yticks([])

#%%
transformed_boundary = base_stack[0]
rho, theta = imp.cart2polar(transformed_boundary[:, 0], transformed_boundary[:, 1])
xnew, ynew = imp.pol2cart(rho, theta + np.pi)
transformed_boundary = np.array([xnew, ynew]).T


