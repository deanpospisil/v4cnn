# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:06:23 2016

@author: dean
"""

net = '/data/dean_data/net_stages/_ref_iter_0.caffemodel'
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(1000))
da = cf.get_net_resp(base_image_nm, ann_dir, iter_name.split('stages/')[1].split('.')[0],
                     stim_trans_cart_dict, stim_trans_dict, require_provenance=True)