
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
resp_dir = '/dean_temp/'

response_file = resp_dir  + 'data/responses/bvlc_reference_caffenet_nat_image_resp_371.nc'

base_image_nm = 'natural'
net = '/home/dean/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet'
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(371))
da = cf.get_net_resp(base_image_nm, ann_dir,
                     net.split('net/')[1].split('.')[0],
                     stim_trans_cart_dict,
                     stim_trans_dict, 
                     require_provenance=False,
                     use_boundary=False,
                     deploy='deploy_fixing_relu_saved.prototxt')
ds = da.to_dataset(name='resp')
ds.to_netcdf(response_file)
print('done')
