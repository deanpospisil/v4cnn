# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:45:48 2016

@author: dean
"""

import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')

import xarray as xr
import apc_model_fit as ac
import pandas as pd
import cPickle as pk


def da_coef_var(da):
    #take xarray and return coefficient of variation
    #expects shapes X unit
    da = da.transpose('shapes', 'unit')
#    da_min_resps = da.min('shapes')
#    lessthanzero = da_min_resps<0
#    if any(lessthanzero):
#        da[:, da_min_resps<0] = da[:, da_min_resps<0] - da_min_resps[da_min_resps<0]
    mu = da.mean('shapes')
    sig = da.reduce(np.std, dim='shapes')
    return 1./(((sig/mu)**2)+1)

def kurtosis(da):
    #take xarray and return coefficient of variation
    #expects shapes X unit
    da = da.chunk({'shapes':370})
    da = da/da.vnorm(('shapes'))
    da = da.transpose('shapes', 'unit')
    mu = da.mean('shapes')
    k = ((da - mu)**4).sum('shapes')
    return k

def ill_conditioned_trans(da):
    da = da.transpose('shapes','x', 'unit')
    da_ms = (da - da.mean('shapes'))**2.
    var_x = da_ms.sum('shapes')
    var_shapes = da_ms.sum('x')
    tot_var = var_shapes.sum('shapes')
    max_frac_var_x = var_x.max('x')/tot_var
    max_frac_var_shapes = var_shapes.max('shapes')/tot_var
    return max_frac_var_shapes, max_frac_var_x

def ill_conditioned_shape_resp(da):
    dmod = xr.DataArray(np.eye(da.shapes.shape[0]), dims=['shapes', 'models'])
    dmod = dmod.chunk({'shapes':370})
    da = da.chunk({'shapes':370})
    eye_r2 = ac.cor_resp_to_model(da, dmod, fit_over_dims=None, prov_commit=False)
    return eye_r2


def take_intersecting_1d_index(indexee, indexer):
    drop_dims = set(indexer.dims) - set(indexee.dims)
    keep_dims = set(indexee.dims) & set(indexer.dims)
    new_coords = indexer.coords.merge(indexer.coords).drop(drop_dims)
    new_dims = ([d for d in indexer.dims if d in keep_dims])

    return xr.DataArray(np.squeeze(indexee.values), new_coords, new_dims)

def translation_invariance(da):
    da = da.transpose('unit', 'x', 'shapes')

    da_ms = (da - da.mean(['shapes'])).squeeze()
    if da_ms.isnull().sum()>0:
    	no_na = [unit.dropna('shapes', how='all').dropna('x', how='all') for unit in da_ms ]
    else:
	no_na = [unit for unit in da_ms]
    s = [np.linalg.svd(unit.transpose('shapes', 'x').values, compute_uv=0) for unit in no_na]
    best_r = np.array([(asingval[0]**2)/(sum(asingval**2)) for asingval in s])

    ti = xr.DataArray(np.squeeze(best_r), dims='unit')
    ti = take_intersecting_1d_index(ti, da)

    return ti

results_folder = top_dir + 'data/an_results/reference/'
cnn_names = [
'APC362_scale_1_pos_(-7, 7, 15)_ref_iter_0',
'APC362_scale_0.45_pos_(-7, 7, 15)_ref_iter_0',
'APC362_scale_0.45_pos_(-50, 48, 50)_ref_iter_0',
'APC362_scale_1_pos_(-50, 48, 50)_ref_iter_0',
]
v4_name = 'V4_362PC2001'

small_run = False
nunits = 10

do_spar = True
do_k = True
do_ti = True
do_eye_r2 = True
do_trans_ill_cond = True
do_apc = True



for cnn_name in cnn_names:

    print(cnn_name)
    #load v4 data
    #load alex data
    v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
    v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
    alex_resp = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp'].load()
    alex_resp_0 = alex_resp.sel(x=0).squeeze()
    if small_run:
        alex_resp_0  = alex_resp_0[:,:nunits]
    # get index for pandas
    keys = [key for key in alex_resp_0['unit'].coords.keys()
            if not alex_resp_0['unit'].coords[key].values.shape==() and key!='unit']
    keys = ['layer_label', 'layer_unit']
    coord = [alex_resp_0['unit'].coords[key].values
            for key in keys]
    index = pd.MultiIndex.from_arrays(coord, names=keys)


    save_folder = top_dir + 'data/an_results/reference/'
    if do_spar:
        #########################
        #coefficient of variation
        print('spar')
        v4_coef_var = da_coef_var(v4_resp_apc.load().copy())
        alex_coef_var = da_coef_var(alex_resp_0.load().copy())

        pd.DataFrame({'coef_var':alex_coef_var}, index=index).to_pickle(save_folder + 'coef_var' + cnn_name)
        pd.DataFrame({'coef_var':v4_coef_var}).to_pickle(save_folder +'coef_var' + v4_name)

    if do_k:
        #########################
        #kurtosis
        print('k')
        v4_k = kurtosis(v4_resp_apc.load().copy())
        alex_k = kurtosis(alex_resp_0.load().copy())

        pd.DataFrame({'k':alex_k}, index=index).to_pickle(save_folder + 'k_'+ cnn_name)
        pd.DataFrame({'k':v4_k}).to_pickle(save_folder + 'k_'+ v4_name)

    if do_ti:
        #translation invariance
        print('ti')
        v4_resp_ti = xr.open_dataset(top_dir + 'data/responses/v4_ti_resp.nc')['resp'].load()

        ti_v4 = translation_invariance(v4_resp_ti)
        if small_run:
            alex_resp = alex_resp.load().squeeze()[:, :, :nunits]
        ti_alex = translation_invariance(alex_resp.load().copy().squeeze())

        pd.DataFrame({'ti':ti_alex}, index=index).to_pickle(save_folder + 'ti_' +  cnn_name)
        pd.DataFrame({'ti':ti_v4}).to_pickle(save_folder + 'ti_' + v4_name)

    if do_trans_ill_cond:
        # translation ill conditioned
        max_frac_var_shapes, max_frac_var_x = ill_conditioned_trans(alex_resp.load().copy().squeeze())
        pd.DataFrame({'shapes_frac_var':max_frac_var_shapes,'x_frac_var':max_frac_var_x},
                     index=index).to_pickle(save_folder + 'trans_ill_cond_' +  cnn_name)

    if do_eye_r2:
        #frac variance explained by identity
        print('eye_r2')
        alex_eye_r2 = ill_conditioned_shape_resp(alex_resp_0.load().copy())
        v4_eye_r2 = ill_conditioned_shape_resp(v4_resp_apc.load().copy())

        pd.DataFrame({'eye_r2':alex_eye_r2}, index=index).to_pickle(save_folder + 'eye_r2_' +  cnn_name)
        pd.DataFrame({'eye_r2':v4_eye_r2}).to_pickle(save_folder + 'eye_r2_' +  v4_name)

    ############
    #APC measurement
    if do_apc:
        print('apc')
        with open(top_dir + 'data/models/PC370_params.p', 'rb') as f:
            shape_dict_list = pk.load(f)
        shape_id = v4_resp_apc.coords['shapes'].values
        shape_dict_list = [shape_dict_list[sn] for sn in shape_id.astype(int)]

        maxAngSD = np.deg2rad(171); minAngSD = np.deg2rad(23)
        maxCurSD = 0.98; minCurSD = 0.09;
        nMeans = 16; nSD = 16
        model_name = 'apc_models_362_16x16'
        fn = top_dir + 'data/models/' + model_name + '.nc'
        dam = ac.make_apc_models(shape_dict_list, shape_id, fn, nMeans, nSD,
                                 maxAngSD, minAngSD, maxCurSD, minCurSD,
                                 prov_commit=False, save=True, replace_prev_model=False)['resp']

        dam_n = dam.copy()
        #shuffle columns
        _ = dam_n.values
        for ind in range(_.shape[1]):
            np.random.shuffle(_[:,ind])

        null_cor_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'unit':100, 'shapes':370}),
                                             dam_n.chunk({'models':1000, 'shapes':370}),
                                            fit_over_dims=None, prov_commit=False)
        alt_cor_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'unit':100, 'shapes':370}),
                                             dam.chunk({'models':1000, 'shapes':370}),
                                            fit_over_dims=None, prov_commit=False)

        null_cor_alex = ac.cor_resp_to_model(alex_resp_0.chunk({'unit':100, 'shapes':370}),
                                             dam_n.chunk({'models':1000, 'shapes':370}),
                                            fit_over_dims=None, prov_commit=False)
        alt_cor_alex = ac.cor_resp_to_model(alex_resp_0.chunk({'unit':100, 'shapes':370}),
                                             dam.chunk({'models':1000, 'shapes':370}),
                                            fit_over_dims=None, prov_commit=False)

        alex_apc_alt = pd.DataFrame({'cor': alt_cor_alex,
                                 'm_cur': alt_cor_alex.coords['cur_mean'].values,
                                 'sd_cur': alt_cor_alex.coords['cur_sd'].values,
                                 'm_or': np.rad2deg(alt_cor_alex.coords['or_mean'].values),
                                 'sd_or': np.rad2deg(alt_cor_alex.coords['or_sd'].values)}, index=index)
        alex_apc_null = pd.DataFrame({'cor': null_cor_alex,
                                 'm_cur':null_cor_alex.coords['cur_mean'].values,
                                 'sd_cur': null_cor_alex.coords['cur_sd'].values,
                                 'm_or': np.rad2deg(null_cor_alex.coords['or_mean'].values),
                                 'sd_or':np.rad2deg(null_cor_alex.coords['or_sd'].values)}, index=index)

        v4_apc_alt = pd.DataFrame({'cor': alt_cor_v4,
                                 'm_cur': alt_cor_v4.coords['cur_mean'].values,
                                 'sd_cur': alt_cor_v4.coords['cur_sd'].values,
                                 'm_or': np.rad2deg(alt_cor_v4.coords['or_mean'].values),
                                 'sd_or': np.rad2deg(alt_cor_v4.coords['or_sd'].values)})
        v4_apc_null = pd.DataFrame({'cor': null_cor_v4,
                                 'm_cur':null_cor_v4.coords['cur_mean'].values,
                                 'sd_cur': null_cor_v4.coords['cur_sd'].values,
                                 'm_or': np.rad2deg(null_cor_v4.coords['or_mean'].values),
                                 'sd_or':np.rad2deg(null_cor_v4.coords['or_sd'].values)})
        apc_alex_name = top_dir + 'data/an_results/reference/apc_' + cnn_name
#        pk.dump({'alex_all_measures':alex_all_measures, 'alex_apc_alt':alex_apc_alt, 'alex_apc_null':alex_apc_null},
#            open(v4ness_alex_name, 'wb'))
        apc_v4_name = top_dir + 'data/an_results/reference/apc_' + v4_name
#        pk.dump({'v4_apc_alt':v4_apc_alt, 'v4_apc_null':v4_apc_null}, open(apc_v4_name, 'wb'))
        pd.concat([v4_apc_alt, v4_apc_null],
                  axis=1, keys=['alt','null']).to_pickle(apc_v4_name)
        pd.concat([alex_apc_alt, alex_apc_null],
                  axis=1, keys=['alt','null']).to_pickle(apc_alex_name)

#    ###########################
#    #organize and save analysis
#    print('save')
#
##    alex_all_measures = pd.DataFrame({'spar':alex_coef_var, 'ti':ti_alex, 'k':alex_k,
##                  'alt_cor':alt_cor_alex, 'null_cor':null_cor_alex}, index=index)
##
##    v4_spar = pd.DataFrame({'spar':v4_coef_var, 'k':v4_k})
##    v4_ti = pd.DataFrame({'ti':ti_v4})
##
##
##    v4ness_alex_name = top_dir + 'data/an_results/reference/v4ness_' + cnn_name + '.p'
##    pk.dump({'alex_all_measures':alex_all_measures, 'alex_apc_alt':alex_apc_alt, 'alex_apc_null':alex_apc_null},
##            open(v4ness_alex_name, 'wb'))
##
##    v4ness_v4_name = top_dir + 'data/an_results/reference/v4ness_' + v4_name + '.p'
##    pk.dump({'v4_apc_alt':v4_apc_alt, 'v4_apc_null':v4_apc_null, 'v4_spar':v4_spar, 'v4_ti':v4_ti},
##            open(v4ness_v4_name, 'wb'))
#
##    pk.load(open(v4ness_alex_name, 'rb'))
##    pk.load(open(v4ness_v4_name, 'rb'))
