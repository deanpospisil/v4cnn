# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:40:24 2015

@author: deanpospisil
"""

import numpy as np
from collections import OrderedDict as ordDict

import os
import sys
import warnings

#make the working directory two above this one
top_dir = os.getcwd().split('net_code')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir +'common')

sys.path.append('/home/dean/caffe/python')

import d_img_process as imp
import d_misc as dm
import pickle
import xarray as xr

def net_imgstack_response(net, stack):
    #stack is expected to be nImages x RGB x rows x cols

    if not net.blobs['data'].data.shape[1:] == stack.shape[1:]:
        warnings.warn('Images are not the correct shape. Input shape: '
        + str(stack.shape[1:]) + ' needed shape: ' + str(net.blobs['data'].data.shape[1:])
        + '. Assuming you just put in grey scale' )

        stack = np.tile(stack, (3,1,1,1))
        stack = np.swapaxes(stack, 0, 1)

    layer_names = [k for k in net.blobs.keys()]

    #shape the data layer, (first layer) to the input
    net.blobs[ layer_names[0] ].reshape(*tuple([stack.shape[0],]) + net.blobs['data'].data.shape[1:])
    net.blobs[ layer_names[0] ].data[... ]= stack
    net.forward()

    all_layer_resp = []
    layer_names_sans_data = layer_names[1:]
    for layer_name in  layer_names_sans_data:

        layer_resp = net.blobs[layer_name].data

        if len(layer_resp.shape)>2:#ignore convolutional repetitions, just pulling center.
            mid = [ round(m/2) for m in np.shape(net.blobs[layer_name].data)[2:]   ]
            layer_resp = layer_resp[ :, :, mid[0], mid[1]]

        all_layer_resp.append(layer_resp)
    response = np.hstack( all_layer_resp )

    return response


def get_indices_for_net_unit_vec(net, layer_names = None):
    #this gives you a couple different indices for describing units in the net.
    if layer_names is None:
        layer_names = [ k for k in net.blobs.keys()][1:]#not including first layer, (data)

    layer_nunits = np.hstack([ net.blobs[layer_name].data.shape[1] for layer_name in  layer_names])

    layer_unit_ind =  np.hstack([range(i) for i in layer_nunits ])

    layer_ind = np.hstack( [ np.ones( layer_nunits[ i ] )*i for i in range( len( layer_nunits ) ) ] )
    resp_descriptor_dict = {}
    resp_descriptor_dict['layer_names'] = layer_names
    resp_descriptor_dict['layer_nunits'] = layer_nunits
    resp_descriptor_dict['layer_unit_ind'] = layer_unit_ind
    resp_descriptor_dict['layer_ind'] = layer_ind

    return resp_descriptor_dict

def identity_preserving_transform_resp(img_stack, stim_trans_cart_dict, net, nimgs_per_pass=260):
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
            stim_trans_cart_dict_sect[key] = stim_trans_cart_dict[key][ stack_ind[0] : stack_ind[1] ]

        #produce those images
        trans_img_stack = imp.imgStackTransform( stim_trans_cart_dict_sect, img_stack )

        #run then and append them
        all_net_resp.append(net_imgstack_response( net, trans_img_stack ))

    #stack up all these responses
    response = np.vstack(all_net_resp)


    return response

def stim_trans_generator(shapes = None, blur= None, scale = None,
                               x = None, y = None, rotation = None):
#takes descrptions of ranges for different transformations (start, stop, npoints)
#produces a cartesian dictionary of those.
    stim_trans_dict = ordDict()
    if not shapes is None:
        stim_trans_dict['shapes'] = np.array(shapes, dtype=float)
    if not blur is None :
        if isinstance(blur,tuple):
            stim_trans_dict['blur'] = np.linspace(*blur)
        else:
            stim_trans_dict['blur'] = blur
    if not scale is None :
        stim_trans_dict[ 'scale' ] = np.linspace(*scale)
    if not x is None :
        stim_trans_dict['x'] = np.linspace(*x)
    if not y is None :
        stim_trans_dict['y'] = np.linspace(*y)
    if not rotation is None :
        stim_trans_dict['rotation'] = np.linspace(*rotation)
    # get all dimensions, into a dict
    stim_trans_cart_dict = dm.cartesian_prod_dicts_lists( stim_trans_dict )
    return stim_trans_cart_dict, stim_trans_dict

def load_npy_img_dirs_into_stack(img_dir):
    #given a directory, loads all the npy images in it, into a stack.
    stack_descriptor_dict = {}
    img_names = dm.load_sorted_dir_numbered_fnms_with_particular_extension( img_dir , 'npy')

    #will need to check this for color images.
    stack_descriptor_dict['img_paths'] = [ img_dir + img_name for img_name in img_names ]
    stack = np.array([ np.load( full_img_name ) for full_img_name in stack_descriptor_dict['img_paths'] ], dtype = float)

    #to do, some descriptor of the images for provenance: commit and input params for base shape gen
    #stack_descriptor_dict['base_shape_gen_inputs'] = [ img_dir + img_name for img_name in img_names ]
    return stack, stack_descriptor_dict


def net_resp_2d_to_xray_nd(net_resp, stim_trans_dict, indices_for_net_unit_vec):
    '''your net_resp will be in 2-d (unitXimage), this function takes into acount
    the stim_trans_dict to organize your responses according to the dimensions
    transformation
    '''
    #get the dimensions of the stimuli in order.
    dims = tuple([len( stim_trans_dict[key]) for key in stim_trans_dict]) + tuple([net_resp.shape[1],])
    #reshape into net_resp_xray
    #this working is dependent on cartesian producing A type cartesian
    #(last index element changes fastest)
    net_resp_xray_vals = np.reshape(net_resp, dims)
    net_dims = [key for key in stim_trans_dict] + ['unit',]
    net_coords =[stim_trans_dict[key] for key in stim_trans_dict] + [range(dims[-1])]
    da = xr.DataArray(net_resp_xray_vals, coords=net_coords, dims=net_dims)

    # adding extra coordinates using indices_for_net_unit_vec
    d = indices_for_net_unit_vec
    da['layer'] = ('unit', d['layer_ind'])
    da['layer_unit'] = ('unit', d['layer_unit_ind'])
    layer_label = [d['layer_names'][int(layer_num)] for layer_num in d['layer_ind']]
    da['layer_label'] = ('unit', layer_label)

    return da

def get_net_resp(base_image_nm, ann_dir, ann_fn, stim_trans_cart_dict,
                 stim_trans_dict, require_provenance=True):
    img_dir = top_dir+'/images/baseimgs/'+base_image_nm+'/'
    base_stack, stack_desc = imp.load_npy_img_dirs_into_stack(img_dir)
    dir_filenames = os.listdir(img_dir)
    #get the current sha from the file
    for name in dir_filenames:
        if 'sha1.pickle' in name:
            with open(img_dir + name, 'rb') as f:
                image_sha = pickle.load(f)

    import caffe
    caffe.set_mode_gpu()
    net = caffe.Net(ann_dir + 'deploy.prototxt', ann_dir + ann_fn + '.caffemodel', caffe.TEST)

    net_resp = identity_preserving_transform_resp(base_stack, stim_trans_cart_dict, net)
    indices_for_net_unit_vec = get_indices_for_net_unit_vec(net)
    da = net_resp_2d_to_xray_nd(net_resp, stim_trans_dict, indices_for_net_unit_vec)

    if require_provenance == True:
        #commit the state of the directory and get is sha identification
        sha = dm.provenance_commit(top_dir)
        da.attrs['resp_sha'] = sha
        da.attrs['img_sha'] = image_sha

    return da

def get_net_resp_name(stim_trans_dict, ann_fn, base_image_nm):
    #base resp name off of net
    xray_desc_name = ''
    for key in stim_trans_dict:
        xray_desc_name = xray_desc_name +'_'+ key +'_'+  str(np.min(stim_trans_dict[key])) \
        +'_'+ str(np.max(stim_trans_dict[key])) +'_'+  str(len(stim_trans_dict[key]))

    xray_desc_name = ann_fn + base_image_nm + xray_desc_name + '.nc'
    return xray_desc_name

