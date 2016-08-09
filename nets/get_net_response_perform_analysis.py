# this is for getting responses across a bunch of nets for a set of stimuli.
#name the directory

import caffe_net_response as cf
ann_dir = '/data/dean_data/net_stages/'
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

def da_coef_var(da):
    da_min_resps = da.min('shapes')
    da[:,da_min_resps<0] = da[:,da_min_resps<0] - da_min_resps[da_min_resps<0]
    mu = da.mean('shapes')
    sig = da.reduce(np.std, dim='shapes')
    return 1./(((mu/sig)**2)+1)

def take_intersecting_1d_index(indexee, indexer):

    drop_dims = set(indexer.dims) - set(indexee.dims)
    keep_dims = set(indexee.dims) & set(indexer.dims)
    new_coords = indexer.coords.merge(indexer.coords).drop(drop_dims)
    new_dims = ([d for d in indexer.dims if d in keep_dims])

    return xr.DataArray(np.squeeze(indexee.values), new_coords, new_dims)
#choose a library of images
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]

ann_fn = 'caffenet_train_iter_'

get_translation_invariance = False
fit_apc_model = False
get_sparsity = False

#all_iter = dm.list_files(ann_dir + '*_iter*.caffe*')
#
##get iterations in order
#iter_numbers = [int(re.findall('\d+', line)[-1]) for line in all_iter]
#all_iter = [all_iter[sort_i] for sort_i in np.argsort(iter_numbers)]
#subset = [len(all_iter)-1, 0]
#all_iter = [all_iter[ind] for ind in subset]

#save_inds = [0, len('all_iter')-1]
#all_iter = ['/data/dean_data/net_stages/_iter_450000.caffemodel',
#'/data/dean_data/net_stages/_ref_iter_0.caffemodel',
#'/data/dean_data/net_stages/_iter_1.caffemodel']

all_iter = ['/home/dean/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
            '/data/dean_data/net_stages/_ref_iter_0.caffemodel']
save_inds = range(0, len(all_iter))

img_n_pix = 227
max_pix_width = [24., 32., 48.]
#boundaries = boundaries * (max_pix_width/biggest_x_y_diff(boundaries))
#biggest_diff = biggest_x_y_diff(boundaries)
#boundaries = boundaries + img_n_pix/2.
mat = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')
s = np.array(mat['shapes'][0])
boundaries = imp.center_boundary(s)
#just save this as pickle.

scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(-1, 370)
center_image = round(img_n_pix/2)
x = (center_image-25, center_image+25, 51)
y = (center_image, center_image, 1)
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids,
                                                                scale=scale,
                                                                x=x,
                                                                y=y)

for i, iter_name in enumerate(reversed(all_iter)):
    print('Total Progress')
    print(i/float(len(all_iter)))

    #get response and save
    iter_subname = iter_name.split('/')[-1].split('.')[0]
    #iteration_number = int(iter_name.split('iter_')[1].split('.')[0])
    response_description = 'APC362_maxpixwidth_' + str(max_pix_width) + '_pos_' + str(x) + iter_subname + '.nc'
    response_file = ('/data/dean_data/responses/' + response_description)

    ti_name = top_dir + 'data/an_results/ti_'+ response_description
    fit_apc_model_name = top_dir + 'data/an_results/apc_'+ response_description
    sparsity_name = top_dir + 'data/an_results/sparsity_'+response_description

    not_all_files_made = not all([os.path.isfile(ti_name), os.path.isfile(fit_apc_model_name), os.path.isfile(sparsity_name)])
    if  not os.path.isfile(response_file) and not_all_files_made:
        da = cf.get_net_resp(base_image_nm,
                             ann_dir,
                             iter_name.split('stages/')[1].split('.')[0],
                             stim_trans_cart_dict,
                             stim_trans_dict,
                             require_provenance=True,
                             use_boundary=True)
        #da.attrs['train'] = iteration_number
        ds = da.to_dataset(name='resp')
        ds.to_netcdf(response_file)

    elif not_all_files_made:
        da = xr.open_dataset(response_file, chunks={'unit':100,'shapes': 370}  )['resp']

    if get_translation_invariance and not os.path.isfile(ti_name):

        da_ms = (da - da.mean(['shapes'])).squeeze()

        s = np.linalg.svd(da_ms.values.T, compute_uv=0)
        best_r_alex = np.array([(asingval[0]**2)/(sum(asingval**2)) for asingval in s])

        ti = xr.DataArray(np.squeeze(best_r_alex), dims='unit')
        ti = take_intersecting_1d_index(ti, da)
        #ti.attrs['resp_coords'] = da_ms.coords.values
        ti.to_dataset(name='tin').to_netcdf(ti_name)

    if not_all_files_made:
        da = da.sel(x=0, method='nearest').squeeze().chunk({'unit':50,'shapes': 370})

    if fit_apc_model and not os.path.isfile(fit_apc_model_name):
        #apc model fit
        if 'dmod' not in locals():
            dmod = xr.open_dataset(top_dir + 'data/models/apc_models_362_16X16.nc')['resp']
        cor = ac.cor_resp_to_model(da, dmod.copy().chunk({'models': 500, 'shapes': 370}))
        #cor.attrs['resp_coords'] = da.coords.values
        cor.to_dataset(name='r').to_netcdf(fit_apc_model_name)

    if get_sparsity and not os.path.isfile(sparsity_name):
        sparsity = da_coef_var(da.load().copy())
        #sparsity.attrs['resp_coords'] = da.coords.values
        sparsity.to_dataset(name='spar').to_netcdf(sparsity_name)

    if i not in save_inds and os.path.isfile(response_file):
        os.remove(response_file)
