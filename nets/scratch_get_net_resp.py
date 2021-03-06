#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:17:58 2017

@author: dean
"""

# this is for getting responses across a bunch of nets for a set of stimuli.
#name the directory

import os
import sys
import re
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0] 
sys.path.append(top_dir+ 'v4cnn')
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')
sys.path.append( top_dir + 'nets/')

import caffe_net_response as cf
import d_misc as dm
import xarray as xr
import apc_model_fit as ac
import d_curve as dc
import d_img_process as imp
import scipy.io as l


#%%
ann_dir = '/home/dean/caffe.orig/models/bvlc_reference_caffenet/'
response_folder = '/loc6tb/data/responses/'
#response_folder = '/dean_temp/data/responses/'
baseImageList = ['PC370', 'formlet']
base_image_nm = 'imgnet_masked'
base_image_nm = baseImageList[0]


all_iter = [
'bvlc_reference_caffenet',
#'blvc_caffenet_iter_1',
]
#base_name = 'bvlc_caffenet_reference_shuffle_layer_'
#all_iter += [base_name+str(layer) for layer in range(7)]
#all_iter = [base_name+str(layer) for layer in [7,]]

deploys = [
'deploy_fixing_relu_saved.prototxt',
]

if len(all_iter)>1:
    deploys = deploys*len(all_iter)
else:
    all_iter = all_iter*len(deploys)

img_n_pix = 227
max_pix_width = [ 32.,]

mat = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')
s = np.array(mat['shapes'][0])

#s = np.load(top_dir + 'img_gen/dp_ang_pos_verts.npy')
base_stack = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(base_stack)
shape_ids = range(-1, 370)
center_image = round(img_n_pix/2.)
y = (64, 164, 51)
#x = (center_image, center_image, 1)
y = x
amp = (100, 255, 2)
#offsetsx = np.array(list(max_pix_width*np.array([0.5, 1, 2])))
shape_ids = np.arange(-1, 370, 1)

scale = max_pix_width/dc.biggest_x_y_diff(base_stack)

stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                     scale=scale,
                     x=x,
                     y=y,
                     amp = amp)



#%%
for  iter_name, deploy  in zip(all_iter, deploys):
    print(iter_name)
    #iteration_number = int(iter_name.split('iter_')[1].split('.')[0])   
    response_description = (iter_name+ 'pix_width'+ str(scale)
                            + '_x_' + str(x) + '_y_' + str(y)
                            + str(base_image_nm)  +'.nc')
    response_file = (response_folder + response_description)

    if  os.path.isfile(response_file):
        print('file already written')
    else:
        da = cf.get_net_resp(base_stack,
                             ann_dir,
                             iter_name,
                             stim_trans_cart_dict,
                             stim_trans_dict,
                             require_provenance=False,
                             use_boundary=True,
                             deploy=deploy)

        da.to_dataset(name='resp').to_netcdf(response_file)




