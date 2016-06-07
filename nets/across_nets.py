# this is for getting responses across a bunch of nets for a set of stimuli.
#name the directory

import caffe_net_response as cf
ann_dir = '/data/dean_data/net_stages/'
import os
import sys
top_dir = os.getcwd().split('v4cnn')[0] 
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
sys.path.append( top_dir + 'common')
top_dir = top_dir + 'v4cnn/'
import d_misc as dm
import xarray as xr

#choose a library of images
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]

ann_fn = 'caffenet_train_iter_'

#

trans= [(-7, 7, 15), (-7, 7, 15), (-50, 48, 50), (-50, 48, 50)]
scales = [1, 0.45, 1, 0.45]

for x, scale in zip(trans,scales):
    stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(370),
                                                         blur=None,
                                                         scale=(scale, scale,1),
                                                         x=x,
                                                         y=None,
                                                         rotation = None)

    all_iter = dm.list_files(ann_dir + '_iter*.caffe*')
    
    for fn in all_iter:
    
        da = cf.get_net_resp(base_image_nm, ann_dir, fn.split('stages/')[1].split('.')[0], 
                             stim_trans_cart_dict, stim_trans_dict, require_provenance=True)
        tn = int(fn.split('iter_')[1].split('.')[0])   
        da.attrs['train'] = tn
        ds = da.to_dataset(name='resp')
        ds.to_netcdf('/data/dean_data/responses/iter_' +str(scale)+'_'+ str(x)+ '_'+ str(tn) + '.nc')
    
