# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:35:38 2016

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:23:36 2016

plots of v4 APC params

@author: dean
"""
import copy
import itertools
from sklearn.utils.extmath import cartesian
import copy
import scipy.io as  l
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
import xarray as xr

import pickle
import apc_model_fit as ac

def apc370models(nMeans=10, nSD=10, perc=5):
    #the parameters of the shapes

    mat = l.loadmat(top_dir + 'data/models/PC2001370Params.mat')
    s = mat['orcurv'][0]

    #adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321]
    a = np.hstack((range(14), range(18,318)))
    a = np.hstack((a, range(322, 370)))
    s = s[a]


    nStim = np.size(s,0)

    angularPosition = []
    curvature = []
    paramLens = []

    for shapeInd in range(nStim):
        angularPosition.append(s[shapeInd][:, 0])
        curvature.append(s[shapeInd][:, 1])
        paramLens.append(np.size(s[shapeInd],0))

    angularPosition = np.array(list(itertools.chain.from_iterable(angularPosition)))
    angularPosition.shape = (np.size(angularPosition),1)

    curvature = np.array(list(itertools.chain.from_iterable(curvature)))
    curvature.shape = (np.size(curvature),1)

    #variable section length striding
    inds = np.empty((2,np.size(paramLens)),dtype = np.intp)
    inds[1,:] = np.cumsum(np.array(paramLens), dtype = np.intp) #ending index
    inds[0,:] = np.concatenate(([0,], inds[1,:-1])) #beginning index

    maxAngSD = np.deg2rad(171)
    minAngSD = np.deg2rad(23)
    maxCurSD = 0.98
    minCurSD = 0.09

    #make this into a pyramid based on d-prime
    orMeans = np.linspace(0, 2*pi-2*pi/nMeans, nMeans)
    orSDs = np.logspace(np.log10(minAngSD),  np.log10(maxAngSD),  nSD)
    curvMeans = np.linspace(-0.5, 1,nMeans)
    curvSDs = np.logspace(np.log10(minCurSD),  np.log10(maxCurSD),  nSD)
    modelParams = cartesian([orMeans,curvMeans,orSDs,curvSDs])
    nModels = np.size( modelParams, 0)

    a = st.vonmises.pdf(angularPosition, kappa = modelParams[:,2]**-1 , loc =  modelParams[:,0]) #
    b = st.norm.pdf(curvature, modelParams[:,1],  modelParams[:,3])
    temp = a * b

    models = np.empty(( 362, nModels ))

    for shapeInd in range(nStim):
        models[ shapeInd, : ] = np.max( temp[ inds[ 0, shapeInd ] : inds[ 1 , shapeInd ] , : ] ,  axis = 0 )

    models = models - np.mean(models,axis = 0)
    magnitude = np.linalg.norm( models, axis = 0)
    magnitude.shape=(1,nModels)
    models = models / magnitude
    del a,b, temp
    return models, modelParams

def modelFits(resp, models):
    resp = resp - np.mean(resp,axis = 0)
    resp = resp / np.linalg.norm(resp, axis = 0)

    #shuffle(resp)
    cov = np.dot(resp.T, models)

    bestFitInd = np.argmax((cov),1)
    bestr = cov[(range(cov.shape[0]), bestFitInd) ].T
    fits =  modelParams[ bestFitInd, : ]
    return fits, bestr

def degen(daa):
    minfracvar = 0.5
    _ = (daa**2)
    tot_var = _.sum('shapes')
    non_zero = tot_var<1e-8
    just_one_shape = (_.max('shapes')/tot_var)>minfracvar
    degen_inds = just_one_shape + non_zero
    return degen_inds

def apc_model_cors_and_nulls(ds, dmod, remove_degen=False):
    ds_list = []
    dmod = dmod.chunk({'models':1000, 'shapes':370})
    for key in ds.keys():
        daa = ds[key].load().copy()
        if daa.dims[0] == 'unit':#make sure its shape x unit
            daa = daa.T

        if remove_degen:
            degen_inds = degen(daa)
            daa = daa[:,-degen_inds]

        cor = ac.cor_resp_to_model(daa.chunk({'unit':100, 'shapes':370}),
                                   dmod, fit_over_dims=None, prov_commit=False)
        cor.attrs['type'] = daa.attrs['type']    
        shape_ind = range(daa['shapes'].count().values)
        for i in range(daa['unit'].count().values):
            np.random.shuffle(shape_ind)
            daa[:, i] = daa[shape_ind, i].values

        cor_shuf = ac.cor_resp_to_model(daa.chunk({'unit':100, 'shapes':370}),
                           dmod, fit_over_dims=None, prov_commit=False)
        dict_ds = {'real':cor ,'shuf':cor_shuf }
        ds_list.append(dict_ds)
    return ds_list

#open those responses, and build apc models for their shapes
with open(top_dir + 'data/models/PC370_params.p', 'rb') as f:
    shape_dict_list = pickle.load(f)

da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
#mat = l.loadmat(top_dir+'data/responses/V4_370PC2001.mat')
#resp=mat['resp'][0][0].T
#np.random.shuffle(resp)
#da = xr.DataArray(resp, dims=['shapes','unit'])
#da = da1

daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
daa=daa.loc[:, 0, :]#without translation
daa.attrs['type'] = 'AlexNet'
da.attrs['type'] = 'V4'

shape_id = da.coords['shapes'].values
shape_dict_list = [shape_dict_list[sn] for sn in shape_id.astype(int)]
maxAngSD = np.deg2rad(171); minAngSD = np.deg2rad(23)
maxCurSD = 0.98; minCurSD = 0.09;
<<<<<<< HEAD
nMeans = 16; nSD =16


fn = top_dir + 'data/models/' + 'apc_models_362_16X16.nc'
dam = ac.make_apc_models(copy.deepcopy(shape_dict_list), shape_id, fn, nMeans, nSD,
                         maxAngSD, minAngSD, maxCurSD, minCurSD,
                         prov_commit=False, save=True, replace_prev_model=False)

dmod = xr.open_dataset(fn, chunks={'models': 100, 'shapes': 370} )['resp']
ds = {'v4':da, 'cnn':daa}
ds = {'cnn':daa}
#ds_list = apc_model_cors_and_nulls(ds, dmod,remove_degen=True)
#with open(top_dir + 'data/models/ds_list_no_degen.p','wb') as f:
=======
#maxCurSD = 0.98; minCurSD = 0.01
nMeans = 16; nSD = 16
fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dam = ac.make_apc_models(shape_dict_list, shape_id, fn, nMeans, nSD,
                         maxAngSD, minAngSD, maxCurSD, minCurSD,
                         prov_commit=False, save=True, replace_prev_model=True)

#load the models you made, and fit them to the cells responses
#models, modelParams = apc370models(nMeans=10, nSD=10)
#dmod = xr.open_dataset(fn, chunks={'models': 100, 'shapes': 370}  )['resp']
#ds = {'v4':da, 'cnn':daa}
#ds = {'v4':da.copy()}
#ds_list = apc_model_cors_and_nulls(ds, dmod)
#with open(top_dir + 'data/models/ds_list_with_degent.p','wb') as f:
>>>>>>> 3a69c33f41c911cef808b767b45f76f6b09ff58f
#    pickle.dump(ds_list, f)


<<<<<<< HEAD
nMeans = 16; nSD = 10
dam = ac.make_apc_models(copy.deepcopy(shape_dict_list), shape_id, fn, nMeans, nSD,
                         maxAngSD, minAngSD, maxCurSD, minCurSD,
                         prov_commit=False, save=False, replace_prev_model=True)
                         
ds_list = apc_model_cors_and_nulls(ds, dam)
with open(top_dir + 'data/models/degen_16x10.p','wb') as f:
    pickle.dump(ds_list, f)

ds_list = apc_model_cors_and_nulls(ds, dam, remove_degen=True)
with open(top_dir + 'data/models/no_degen_16x10.p','wb') as f:
    pickle.dump(ds_list, f)


nMeans = 16; nSD = 16
dam = ac.make_apc_models(copy.deepcopy(shape_dict_list), shape_id, fn, nMeans, nSD,
                         maxAngSD, minAngSD, maxCurSD, minCurSD,
                         prov_commit=False, save=False, replace_prev_model=True)
                         
ds_list = apc_model_cors_and_nulls(ds, dam)
with open(top_dir + 'data/models/degen_16x16.p','wb') as f:
    pickle.dump(ds_list, f)

ds_list = apc_model_cors_and_nulls(ds, dam, remove_degen=True)
with open(top_dir + 'data/models/no_degen_16x16.p','wb') as f:
    pickle.dump(ds_list, f)





=======
#nMeans = 2; nSD = 1
#dam = ac.make_apc_models(copy.deepcopy(shape_dict_list), shape_id, fn, nMeans, nSD,
#                         maxAngSD, minAngSD, maxCurSD, minCurSD,
#                         prov_commit=False, save=False, replace_prev_model=True)
#ds_list = apc_model_cors_and_nulls(ds, dam)
#with open(top_dir + 'data/models/degen_16x10.p','wb') as f:
#    pickle.dump(ds_list, f)
#
#ds_list = apc_model_cors_and_nulls(ds, dam, remove_degen=True)
#with open(top_dir + 'data/models/no_degen_16x10.p','wb') as f:
#    pickle.dump(ds_list, f)
#
#
#nMeans = 10; nSD = 10
#dam = ac.make_apc_models(copy.deepcopy(shape_dict_list), shape_id, fn, nMeans, nSD,
#                         maxAngSD, minAngSD, maxCurSD, minCurSD,
#                         prov_commit=False, save=False, replace_prev_model=True)
#ds_list = apc_model_cors_and_nulls(ds, dam)
#with open(top_dir + 'data/models/degen_16x16.p','wb') as f:
#    pickle.dump(ds_list, f)
#
#ds_list = apc_model_cors_and_nulls(ds, dam, remove_degen=True)
#with open(top_dir + 'data/models/no_degen_16x16.p','wb') as f:
#    pickle.dump(ds_list, f)
#
#
#
#
#
>>>>>>> 3a69c33f41c911cef808b767b45f76f6b09ff58f



