# this is for getting responses across a bunch of nets for a set of stimuli.
#name the directory

import caffe_net_response as cf
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
import os
import sys
top_dir = os.getcwd().split('net_code')[0] 
sys.path.append(top_dir+ 'net_code/')
sys.path.append( top_dir + 'xarray')
sys.path.append( top_dir + 'common')
top_dir = top_dir + 'net_code/'
import d_misc as dm
import xarray as xr

#choose a library of images
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]

ann_fn = 'caffenet_train_iter_'


stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(370),
                                                             blur=None,
                                                             scale =None,
                                                             x=(-50, 50, 51),
                                                             y=None,
                                                             rotation = None)
all_iter = dm.list_files('/home/dean/caffe/models/bvlc_reference_caffenet/caff*.caffe*')

for fn in all_iter:

    da = cf.get_net_resp(base_image_nm, ann_dir, fn.split('ce_caffenet/')[1].split('.')[0], 
                         stim_trans_cart_dict, stim_trans_dict, require_provenance=True)
    tn = int(fn.split('iter_')[1].split('.')[0])   
    da.attrs['train']= tn
    ds = da.to_dataset(name ='resp')
    ds.to_netcdf(top_dir + 'data/responses/iter_' + str(tn) + '.nc')


