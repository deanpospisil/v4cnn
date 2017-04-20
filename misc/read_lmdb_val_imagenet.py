#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:15:53 2017

@author: dean
"""
import sys
sys.path.append('/home/dean/caffe/python')
import caffe
lmdb_dir = '/home/dean/caffe.orig/examples/imagenet/ilsvrc12_val_lmdb/'

import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt

lmdb_env = lmdb.open(lmdb_dir)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()
images = []
for ((key, value),_) in zip(lmdb_cursor, range(5000)):
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)

    #CxHxW to HxWxC in cv2
    images.append(data)

    #print('{},{}'.format(key, label))
    
images = np.array(images) 
#%%
images_fix = images
images_fix = images_fix.swapaxes(1, -1).swapaxes(1,2)
images_fix = np.stack((images_fix[..., 2], images_fix[..., 1], images_fix[..., 0]), -1)
#for i in range(20):
#    plt.figure(figsize=(5,5))
#    plt.imshow(images_fix[i])
    
#%% make image patches
img_size = np.shape(images)[-1]
x, y = np.meshgrid(np.arange(img_size), np.arange(img_size)) - np.round(img_size/2)
d = (x**2 + y**2)**0.5
sigma = 6
mask = np.exp((-(d)**2)/(2*sigma**2))*1
mask = mask * (d<=16)
#mask = (d<16)
#plt.plot(mask)

m_images_fix = images_fix * mask[np.newaxis, :, :, np.newaxis]
plt.figure()
plt.imshow(m_images_fix[0]/np.max(m_images_fix[0]))
plt.figure()
plt.imshow(images_fix[0])

#%%
m_images = images * mask[np.newaxis, np.newaxis, : , :]
m_images = np.concatenate((np.zeros((1,) + m_images.shape[1:]), m_images)) #add baseline image
# this is for getting responses across a bunch of nets for a set of stimuli.
#name the directory
save_dir = '/home/dean/Desktop/v4cnn/images/baseimgs/imgnet_masked/'
import scipy as sc

for i, an_image in enumerate(m_images[:380]):
    np.save(save_dir + str(i-1), an_image[:, 29:, 29:])
    sc.misc.imsave(save_dir + str(i-1) + '.bmp', an_image.T)
    

##%%
#import caffe_net_response as cf
#import os
#import sys
#import re
#import numpy as np
#top_dir = os.getcwd().split('v4cnn')[0]
#sys.path.append(top_dir+ 'v4cnn/')
#sys.path.append( top_dir + 'xarray')
#sys.path.append( top_dir + 'common')
#top_dir = top_dir + 'v4cnn/'
#import d_misc as dm
#import xarray as xr
#import apc_model_fit as ac
#import d_curve as dc
#import d_img_process as imp
#import scipy.io as l
#
#ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
#response_folder = '/home/dean/Desktop/v4cnn/data/responses/'
##response_folder = '/dean_temp/data/responses/'
#baseImageList = ['PC370', 'formlet']
#base_image_nm = baseImageList[2]
#
#
#all_iter = [
#'bvlc_reference_caffenet',
##'blvc_caffenet_iter_1',
#]
##base_name = 'bvlc_caffenet_reference_shuffle_layer_'
##all_iter += [base_name+str(layer) for layer in range(7)]
##all_iter = [base_name+str(layer) for layer in [7,]]
#
#
#deploys = 'deploy_fixing_relu_saved.prototxt',
#
#
#if len(all_iter)>1:
#    deploys = deploys*len(all_iter)
#else:
#    all_iter = all_iter*len(deploys)
#
#img_n_pix = 227
#max_pix_width = [ 32.,]
#
#mat = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')
#s = np.array(mat['shapes'][0])
#boundaries = imp.center_boundary(s)
#scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
#shape_ids = range(-1, 370)
#center_image = round(img_n_pix/2.)
#y = (center_image-40, center_image+40, 21)
#x = (center_image-40, center_image+40, 21)
#
##y = (center_image, center_image, 11)
#amp = (255, 255, 1)
#amp = None
#stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
#                                                                scale=scale,
#                                                                x=x,
#                                                                y=y,
#                                                                amp=amp)
#for  iter_name, deploy  in zip(all_iter, deploys):
#    print(iter_name)
#    #iteration_number = int(iter_name.split('iter_')[1].split('.')[0])   
#    response_description = (iter_name+ '_APC362_pix_width'+ str(max_pix_width)
#                            + '_x_' + str(x) + '_y_' + str(y) 
#                            +'_amp_'+ str(amp) + '.nc')
#    response_file = (response_folder + response_description)
#
#    if  os.path.isfile(response_file):
#        print('file already written')
#    else:
#        da = cf.get_net_resp(base_image_nm,
#                             ann_dir,
#                             iter_name,
#                             stim_trans_cart_dict,
#                             stim_trans_dict,
#                             require_provenance=False,
#                             use_boundary=True,
#                             deploy=deploy)
#
#        da.to_dataset(name='resp').to_netcdf(response_file)
#
#
#
#
#
#
