# this is for getting responses across a bunch of nets for a set of stimuli.
#name the directory

import caffe_net_response as cf
import os
import sys
import re
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
sys.path.append( top_dir + 'common')
top_dir = top_dir + 'v4cnn/'
import d_misc as dm
import xarray as xr
import apc_model_fit as ac
import d_curve as dc
import d_img_process as imp
import scipy.io as l
import caffe
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
response_folder = '/home/dean/Desktop/v4cnn/data/responses/'
response_folder = '/dean_temp/data/responses/'
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]


all_iter = [
'bvlc_reference_caffenet',
#'blvc_caffenet_iter_1',
]
iter_name = all_iter[0]
ann_fn = iter_name
deploys = [
'deploy_fixing_relu_saved.prototxt',
]
deploy = deploys[0]

img_n_pix = 227
max_pix_width = [ 32.,]

mat = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')
s = np.array(mat['shapes'][0])
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(-1, 370)
center_image = round(img_n_pix/2.)
x = (center_image-25, center_image+25, 51)
#x =  (center_image, center_image, 1)
y = (center_image, center_image, 1)
amp = (255, 255, 1)
amp = None
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                                                                scale=scale,
                                                                x=x,
                                                                y=y,
                                                                amp=amp)

s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
shape_stack = dc.center_boundary(s)
net = caffe.Net(ann_dir + deploy, ann_dir + ann_fn + '.caffemodel', caffe.TEST)

response_description = iter_name+ '_all_conv_APC362_pix_width'+ str(max_pix_width) + '_pos_' + str(x) +'_amp_'+ str(amp) + '.nc'
response_file = (response_folder + response_description)

response = cf.identity_preserving_transform_resp(shape_stack, stim_trans_cart_dict, net, only_middle_conv=False,
                                                 record_up_to_layer=5)
np.save(response_file[:-3], np.array(response))
#da.to_dataset(name='resp').to_netcdf(response_file)




