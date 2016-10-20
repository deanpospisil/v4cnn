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

ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
response_folder = '/home/dean/Desktop/v4cnn/data/responses/'
response_folder = '/dean_temp/data/responses/'
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]


all_iter = [
#'bvlc_reference_caffenet',
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
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(-1, 370)
center_image = round(img_n_pix/2.)
x = (center_image-50, center_image+50, 51)
#x =  (center_image, center_image, 1)
y = (center_image, center_image, 1)
amp = (255, 255, 1)
amp = None
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                                                                scale=scale,
                                                                x=x,
                                                                y=y,
                                                                amp=amp)

for  iter_name, deploy  in zip(all_iter, deploys):
    print(iter_name)
    #iteration_number = int(iter_name.split('iter_')[1].split('.')[0])   
    response_description = iter_name+ 'APC362_pix_width'+ str(max_pix_width) + '_pos_' + str(x) +'_amp_'+ str(amp) + '.nc'

    response_file = (response_folder + response_description)

    if  os.path.isfile(response_file):
        print('file already written')
    else:
        da = cf.get_net_resp(base_image_nm,
                             ann_dir,
                             iter_name,
                             stim_trans_cart_dict,
                             stim_trans_dict,
                             require_provenance=True,
                             use_boundary=True,
                             deploy=deploy)

        da.to_dataset(name='resp').to_netcdf(response_file)




