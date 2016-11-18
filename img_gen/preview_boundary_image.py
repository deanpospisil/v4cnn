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
import d_curve as dc
s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)
import caffe_net_response as cf
import d_img_process as imp

img_n_pix = 227
max_pix_width = [200.,]
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
figure_folder = top_dir + 'analysis/figures/images/'
plt.figure(figsize=(6,12))
center = 113
box_lengths = [11,51,99,131,163]
trans_img_stack =np.array(imp.boundary_stack_transform(stim_trans_cart_dict, base_stack, npixels=227))
#plot smallest and largest shape
no_blank_image = trans_img_stack[1:]
extents = (no_blank_image.sum(1)>0).sum(1)
plt.subplot(211)
plt.imshow(no_blank_image[208],cmap=plt.cm.Greys_r)
plt.xticks([]);plt.yticks([])

plt.savefig(top_dir+'analysis/figures/images/example.jpeg')
'''
plt.imshow(no_blank_image[np.argmax(extents)],
                          interpolation = 'nearest', cmap=plt.cm.Greys_r)

for box_length in box_lengths:
    rectangle = plt.Rectangle((center-np.ceil(box_length/2.), center-np.ceil(box_length/2)),
                               box_length, box_length, fill=False, edgecolor='r')
    plt.gca().add_patch(rectangle)

plt.subplot(212)
plt.imshow(no_blank_image[np.argmin(extents)],
                          interpolation = 'nearest', cmap=plt.cm.Greys_r)
for box_length in box_lengths:
    rectangle = plt.Rectangle((center-np.ceil(box_length/2.), center-np.ceil(box_length/2)),
                               box_length, box_length, fill=False, edgecolor='r')
    plt.gca().add_patch(rectangle)

plt.grid()
plt.xticks([]);plt.yticks([])
plt.savefig(figure_folder + 'widest_narrowest_image' + '.svg')



#for i , shape in enumerate(trans_img_stack[270:370]):
#    plt.subplot(5, 5, i + 1)
#    plt.imshow(shape)
#    plt.xticks([]);plt.yticks([])
'''