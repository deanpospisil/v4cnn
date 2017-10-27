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
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
response_folder = '/loc6tb/data/responses/'
#response_folder = '/dean_temp/data/responses/'
baseImageList = ['PC370', 'formlet']
base_image_nm = 'imgnet_masked'
base_image_nm = baseImageList[0]


all_iter = [
'bvlc_reference_caffenet',
#'bvlc_caffenet_reference_increase_wt_cov_random0.9',
#'bvlc_caffenet_reference_increase_wt_cov_fc6_0.2',
#'bvlc_caffenet_reference_increase_wt_cov_fc6_0.3',
#'bvlc_caffenet_reference_increase_wt_cov_fc6_0.4',
#'bvlc_caffenet_reference_increase_wt_cov_fc6_0.5',
#'bvlc_caffenet_reference_increase_wt_cov_fc6_0.6',
#'bvlc_caffenet_reference_increase_wt_cov_fc6_0.7',
#'bvlc_caffenet_reference_increase_wt_cov_fc6_0.8',
#'bvlc_caffenet_reference_increase_wt_cov_0.5',
#'bvlc_caffenet_reference_increase_wt_cov_0.75',
#'bvlc_caffenet_reference_increase_wt_cov_0.95'
'blvc_caffenet_iter_1',
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
#scale = None
shape_ids = range(-1, 9)
center_image = round(img_n_pix/2.)
#y = (center_image-80, center_image+80, 21)
x = (32, 196, 83)
x = (64, 164, 52)

#x = (center_image-80, center_image+80, 21)
y = (center_image, center_image, 1)
from collections import OrderedDict as ordDict

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
for  iter_name, deploy  in zip(all_iter, deploys):
    print(iter_name)
    #iteration_number = int(iter_name.split('iter_')[1].split('.')[0])   
    response_description = (iter_name+ 'pix_width'+ str(max_pix_width)
                            + '_x_' + str(x) + '_y_' + str(y) + '_offsets_'
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




