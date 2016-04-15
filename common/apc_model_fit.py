# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:28:57 2016

@author: dean
"""
# analysis
import scipy.stats as st
import numpy as np
import warnings
import os, sys
from collections import OrderedDict as ord_d

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm


def get_2d_dims_right(vec, dims_order=(1, 0)):
    dims = vec.shape
    if len(dims)>2:
        warnings.warn('model params should not have more than 2-d')
        right_dims = None

    elif len(dims) < 2 :
        right_dims = np.expand_dims(vec, axis=dims_order[1])

    elif dims[ dims_order[1]] < dims[ dims_order[0]]:
        right_dims = np.swapaxes( vec, 1, 0)

    return right_dims

#takes a set of points in apc plane and makes prediction based on different receptive fields
def apc_models(shape_dict_list=[{'curvature': None, 'orientation': None} ],
                                model_params_dict={'or_sd': [3.14],
                                                   'or_mean':[3.14],
                                                   'cur_mean':[1],
                                                   'cur_sd':[0.1]}):
     # make sure everything has the right dimensionality for broadcating
    for key in model_params_dict:
        vec = np.array(model_params_dict[key])
        model_params_dict[key] = get_2d_dims_right(vec, dims_order=(1,0))

    # make sure everything has the right dimensionality for broadcating, figure out a more succint way to do this
    for ind, a_shape in enumerate(shape_dict_list):
        for key in a_shape:
            vec = np.array(a_shape[key])
            a_shape[key] = get_2d_dims_right(vec, dims_order= (0,1) )

        shape_dict_list[ind] = a_shape

    #initialize our distributions
    von_rv = st.vonmises( kappa = model_params_dict['or_sd']**-1 , loc = model_params_dict['or_mean'] )
    #von_rv = st.norm( scale = model_params_dict['or_sd'] , loc = model_params_dict['or_mean'] )

    norm_rv = st.norm( scale = model_params_dict['cur_sd'] , loc = model_params_dict['cur_mean'] )

    model_resp = []
    #get responses to all points for each axis ap and c then their product, then the max of all those points as the resp

    for i, apc_points in enumerate(shape_dict_list):#had to break this up per memory issues
        print(i)
        model_resp_all_apc_points = von_rv.pdf(apc_points['orientation']) * norm_rv.pdf( apc_points['curvature'])
        model_resp.append(np.array([np.max(model_resp_all_apc_points, axis=0)]))

    #mean subtract
    model_resp = np.squeeze(np.array(model_resp))
    model_resp = model_resp - np.mean(model_resp, axis = 0 )
    #scale
    magnitude = np.linalg.norm( model_resp, axis = 0)
    model_resp = model_resp / magnitude

    return model_resp

def make_apc_models(shape_dict_list, shape_id, fn, nMeans, nSD,
                          maxAngSD, minAngSD, maxCurSD, minCurSD,
                          model_params_dict=None, prov_commit=False, cart=True,
                          save=False):
    #make this into a pyramid based on d-prime
    fn = top_dir + 'data/models/' + fn

    if cart:
        orMeans = np.linspace(0, 2*np.pi - 2*np.pi / nMeans, nMeans)
        orSDs = np.logspace(np.log10( minAngSD ), np.log10( maxAngSD ), nSD )
        curvMeans = np.linspace( -0.5, 1, nMeans )
        curvSDs = np.logspace( np.log10(minCurSD), np.log10(maxCurSD), nSD )
        model_params_dict = ord_d({'or_sd': orSDs, 'or_mean':orMeans,
                             'cur_mean' :curvMeans, 'cur_sd':curvSDs})
        model_params_dict = dm.cartesian_prod_dicts_lists( model_params_dict )

    if not os.path.isfile(fn):
        model_resp = apc_models(shape_dict_list=shape_dict_list,
                                model_params_dict=model_params_dict)
        dam = xr.DataArray(model_resp, dims = ['shapes', 'models'], coords=[shape_id, range(model_resp.shape[1])])

        for key in model_params_dict.keys():
            dam[key] = ('models', np.squeeze(model_params_dict[key]))

        if prov_commit:
            sha = dm.provenance_commit(top_dir)
            dam.attrs['model'] = sha
        if save:
            ds = xr.Dataset({'resp': dam})
            ds.to_netcdf(fn)

        return dam
    else:
        warnings.warn('Model File has Already Been Written.')
        return xr.open_dataset(fn)

def cor_resp_to_model(da, dmod, fit_over_dims=None, prov_commit=False):
    #typically takes da, data, and dm, a set of linear models, an fn to write to,
    #and finally fit_over_dims which says over what dims is a models fit supposed to hold.
    da = da - da.mean(('shapes'))
    ats = dmod.attrs
    dmod = dmod - dmod.mean(('shapes'))
    dmod = dmod/dmod.vnorm(('shapes'))

    resp_n = da.vnorm(('shapes'))
    proj_resp_on_model = da.dot(dmod)

    if not fit_over_dims == None:
        resp_norm = resp_n.vnorm(fit_over_dims)
        proj_resp_on_model_var = proj_resp_on_model.sum(fit_over_dims)
        n_over = 0
        #count up how many unit vectors you'll be applying for each r.
        for dim in fit_over_dims:
            n_over = n_over + len(da.coords[dim].values)
    else:
        resp_norm = resp_n
        proj_resp_on_model_var = proj_resp_on_model
        n_over = 1

    all_cor = (proj_resp_on_model_var) / (resp_norm * (n_over**0.5))
    all_cor = all_cor.dropna('unit')
    all_cor = all_cor.load()


    corarg = all_cor.argmax('models')
    model_fit_params = dmod.coords['models'][corarg]
    cor = all_cor.max('models')

    for key in model_fit_params.coords.keys():
        cor[key] = ('unit', np.squeeze(model_fit_params[key]))

    if prov_commit==True and ('model' in ats.keys()):
        sha = dm.provenance_commit(top_dir)
        cor.attrs['analysis'] = sha
        cor.attrs['model'] = ats['model']

    return cor


