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

def identity_preserving_transform_resp(shape_stack, stim_trans_cart_dict, net, nimgs_per_pass=150):
    #takes stim_trans_cart_dict, pulls from img_stack and transform accordingly,
    #gets nets responses.

    n_imgs = len( stim_trans_cart_dict[stim_trans_cart_dict.keys()[0]] )
    stack_indices, remainder = dm.sectStrideInds( nimgs_per_pass, n_imgs )

    #now divide the dict up into sects.
    #order doesn't matter using normal dict, imgStackTransform has correct order
    stim_trans_cart_dict_sect = {}
    all_net_resp = []
    for stack_ind in stack_indices:
        print(stack_ind[1]/np.double(stack_indices[-1][1]))

        #load up a chunk of transformations
        for key in stim_trans_cart_dict:
            stim_trans_cart_dict_sect[key] = stim_trans_cart_dict[key][stack_ind[0]:stack_ind[1]]

        #produce those images
        if not 2 in shape_stack[0].shape:#check if it is just 2-d ie a boundary
            trans_img_stack = imp.imgStackTransform(stim_trans_cart_dict_sect, shape_stack)
        else:
            trans_img_stack =np.array(imp.boundary_stack_transform(stim_trans_cart_dict_sect,
                                                           shape_stack, npixels=227))


        #run then and append them
        all_net_resp.append(net_imgstack_response(net, trans_img_stack))

    #stack up all these responses
    response = np.vstack(all_net_resp)


    return response

s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)
import d_curve as dc
import caffe_net_response as cf
import d_img_process as imp

img_n_pix = 227
max_pix_width = [96.,]
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(-1, 370)
center_image = round(img_n_pix/2)
x = (center_image, center_image+48, 1)
y = (center_image, center_image, 1)
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                                                                scale=scale,
                                                                x=x,
                                                                y=y)

trans_img_stack =np.array(imp.boundary_stack_transform(stim_trans_cart_dict, base_stack, npixels=227))
for i , shape in enumerate(trans_img_stack[270:370]):
    plt.subplot(10,10, i + 1)
    plt.imshow(shape)
    plt.xticks([]);plt.yticks([])
