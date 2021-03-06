# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:06:23 2016

@author: dean
"""

import caffe_net_response as cf
ann_dir = '/data/dean_data/net_stages/'
import os
import sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
sys.path.append( top_dir + 'common')
top_dir = top_dir + 'v4cnn/'
resp_dir = top_dir

response_file = resp_dir  + 'data/responses/bvlc_reference_caffenet_wyeth_check.nc'


#img_dir = '/loc6tb/data/images/' + base_image_nm + '/'
#
#base_stack, stack_desc = imp.load_npy_img_dirs_into_stack(img_dir)



base_image_nm = 'Kiani_ImageSet'
net = '/home/dean/caffe.orig/models/bvlc_reference_caffenet/bvlc_reference_caffenet'
ann_dir = '/home/dean/caffe.orig/models/bvlc_reference_caffenet/'



stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(3))
da = cf.get_net_resp(base_image_nm, ann_dir,
                     net.split('net/')[1].split('.')[0],
                     stim_trans_cart_dict,
                     stim_trans_dict, 
                     require_provenance=False,
                     use_boundary=False,
                     deploy='deploy_fixing_relu_saved.prototxt')
                     #deploy='deploy_relu_saved_up_to_conv2.prototxt')
ds = da.to_dataset(name='resp')
ds.to_netcdf(response_file)
print('done')
