#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:08:36 2017

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

import numpy as np
import caffe
import matplotlib.pyplot as plt

sys.path.append('/home/dean/caffe/python')
caffe_root = '/home/dean/caffe/'
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ann_fn = 'bvlc_reference_caffenet'

caffe.set_mode_gpu()

net = caffe.Net(ann_dir + 'deploy_fixing_relu_saved.prototxt', 
                ann_dir + ann_fn + '.caffemodel', caffe.TEST)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
         
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


example_image_dir = '/home/dean/Desktop/v4cnn/images/test_imgs_wyeth/'
img_names = os.listdir(example_image_dir)
img_names = [img_name for img_name in img_names 
             if ('JPEG' in img_name or 'jpg' in img_name)]
img_dir ='/loc6tb/data/images/test_imgs_wyeth/'
for img_ind, an_image in enumerate(img_names):
    image = caffe.io.load_image(example_image_dir + an_image)
    
    ti = transformer.preprocess('data', image)
    np.save(img_dir + str(img_ind) , ti)

    f = open(img_dir+ str(img_ind) +'.txt', 'w')
    for dim in np.shape(ti):
        f.write(str(dim) + ' ')
    f.write('\n')
    for v in ti.flatten():
        f.write(str('{0:.16f}'.format(v)) + ' ')
    f.close()
    
   
    f.close()   
    
#%%
response_file = resp_dir  + 'data/responses/bvlc_reference_caffenet_wyeth_check.nc'


#img_dir = '/loc6tb/data/images/' + base_image_nm + '/'
#
#base_stack, stack_desc = imp.load_npy_img_dirs_into_stack(img_dir)



base_image_nm = 'test_imgs_wyeth'
net = '/home/dean/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet'
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(2))
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
print(da.isel(unit=496))

#%%

import xarray as xr
data_dir = '/loc6tb/'
cnn_names =['bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370',]
if sys.platform == 'linux2':
    da = xr.open_dataset(data_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
else:
    da = xr.open_dataset(data_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp']
da = da.sel(unit=slice(0, None, 1)).squeeze()

da.isel(unit=496).sel(x=114)[1:].plot();plt.savefig(top_dir + 'conv2_112_unit497_apc_resp.pdf')

print(da.isel(unit=496).sel(x=114)[1:3])

middle = np.round(len(da.coords['x'])/2.).astype(int)
da_0 = da.sel(x=da.coords['x'][middle])

indexes = np.unique(da.coords['layer_label'].values, return_index=True)[1]
layer_label = [da.coords['layer_label'].values[index] for index in sorted(indexes)]
   
import pickle as pk
def open_cnn_analysis(fn, layer_label):
    try:
        an=pk.load(open(fn,'rb'), 
                   encoding='latin1')
    except:
        an=pk.load(open(fn,'rb'))
    fvx = an[0].sel(concat_dim='r')
    rf = an[0].sel(concat_dim='rf')
    cnn = an[1]
    return fvx, rf, cnn
fn = 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p'

results_dir = data_dir + '/data/an_results/'
alt = open_cnn_analysis(results_dir +  fn, layer_label)[-1]


#%%
def plot_resp_on_sort_shapes(ax, shapes, resp, top=25, fs=20, shrink=.5, colorbar=False):
    c_imgs = np.zeros(np.shape(shapes) + (4,))
    respsc = (resp - resp.min())
    respsc = respsc/respsc.max()
    
    scale = cm.cool(respsc)
    resp_sort_inds = np.argsort(resp)[::-1]
    
    for i, a_color in enumerate(scale):
        c_imgs[i, np.nonzero(shapes[i])[0], np.nonzero(shapes[i])[1],:] = a_color
    
    im = ax.imshow(np.tile(respsc,(2,1)), cmap=cm.cool, interpolation='nearest')
    if colorbar:
        cbar = ax.get_figure().colorbar(im, ax=ax, shrink=shrink, ticks=[0,1]) 
        cbar.ax.set_ylabel('',rotation='horizontal', fontsize=fs/1.5, ha='center')
        cbar.ax.tick_params(axis='both', which='both',length=0)
#    cbar = ax.get_figure().colorbar(im, ax=ax, shrink=shrink, 
#            ticks=[np.min(respsc), np.max(respsc)], aspect=10)
#    cbar.ax.set_yticklabels([]) 
    #cbar.ax.set_ylabel('Normalized\nResponse', rotation='horizontal', fontsize=fs/1.5, ha='left')
    
    data = vis_square(ax, c_imgs[resp_sort_inds][:top])
    ax.imshow(data, interpolation='nearest')
    #beautify(ax, ['top','right','left','bottom'])
    return data

    
ax = ax_list[ax_ind+1]
no_blank_image = trans_img_stack[1:]
a = np.hstack((range(14), range(18, 318)));a = np.hstack((a, range(322, 370)))
no_blank_image = no_blank_image[a]/255.

data = plot_resp_on_sort_shapes(ax, no_blank_image, dat[0], top=16, fs=fs, 
                                shrink=0.75, colorbar=colorbar)


