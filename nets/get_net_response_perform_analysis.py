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

def da_coef_var(da):
    da_min_resps = da.min('shapes')
    da[:,da_min_resps<0] = da[:,da_min_resps<0] - da_min_resps[da_min_resps<0]
    mu = da.mean('shapes')
    sig = da.reduce(np.std, dim='shapes')
    return 1./(((mu/sig)**2)+1)

#choose a library of images
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]

ann_fn = 'caffenet_train_iter_'

get_translation_invariance = True
fit_apc_model = True
get_sparsity = True

all_iter = dm.list_files(ann_dir + '_iter*.caffe*')

#get iterations in order
iter_numbers = [int(re.findall('\d+', line)[-1]) for line in all_iter]
all_iter = [all_iter[sort_i] for sort_i in np.argsort(iter_numbers)]
subset = [len(all_iter)-1,] + range(0, len(all_iter), 49)
all_iter = [all_iter[ind] for ind in subset]
save_inds = range(0, len(all_iter))

trans_x = [(-7, 7, 15), (-7, 7, 15), (-50, 48, 50), (-50, 48, 50)]
scales = [0.45, 1, 0.45, 1]


for x, scale in zip(trans_x, scales):
    stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(
                                                         shapes=range(370),
                                                         blur=None,
                                                         scale=(scale, scale,1),
                                                         x=x,
                                                         y=None,
                                                         rotation = None)

    for i, iter_name in enumerate(all_iter):
        print('Total Progress')
        print(i/float(len(all_iter)))

        #get response and save
        iteration_number = int(iter_name.split('iter_')[1].split('.')[0])
        response_file = ('/data/dean_data/responses/iter_scale_' + str(scale) +
        '_pos_' + str(x) + '_' + str(iteration_number) + '.nc')
        if  not os.path.isfile(response_file):
            da = cf.get_net_resp(base_image_nm, ann_dir, iter_name.split('stages/')[1].split('.')[0],
                                 stim_trans_cart_dict, stim_trans_dict, require_provenance=True)
            da.attrs['train'] = iteration_number
            ds = da.to_dataset(name='resp')
            ds.to_netcdf(response_file)

            
        da = xr.open_dataset(response_file, chunks={'unit':100,'shapes': 370}  )['resp']

        
        ti_name = top_dir + 'data/an_results/ti_'+ iter_name.split('net_stages/')[1]
        if get_translation_invariance and not os.path.isfile(ti_name):
            
            da_ms = da - da.mean(['shapes'])
            s = np.linalg.svd(da_ms.values.T, compute_uv=0)
            best_r_alex = np.array([(asingval[0]**2)/(sum(asingval**2)) for asingval in s])
            ti = xr.DataArray(best_r_alex).reindex_like(da.sel(x=0, method='nearest'))
            ti.to_dataset(name='ti').to_netcdf(ti_name)

        da = da.sel(x=0, method='nearest').squeeze().chunk({'unit':50,'shapes': 370})
        fit_apc_model_name = top_dir + 'data/an_results/noTI_r_'+ iter_name.split('net_stages/')[1]

        if fit_apc_model and not os.path.isfile(fit_apc_model_name):
            #apc model fit
            if 'dmod' not in locals():
                dmod = xr.open_dataset(top_dir + 'data/models/apc_models_362_16X16.nc')['resp']
            cor = ac.cor_resp_to_model(da, dmod.copy().chunk({'models': 500, 'shapes': 370}))
            cor.to_dataset(name='r').to_netcdf(fit_apc_model_name)
        
        sparsity_name = top_dir + 'data/an_results/sparsity_'+ iter_name.split('net_stages/')[1]       
        if get_sparsity and not os.path.isfile(sparsity_name):
            sparsity = da_coef_var(da.load().copy())
            sparsity.to_dataset(name='spar').to_netcdf(sparsity_name)

        if i not in save_inds:
            os.remove(response_file)
