# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:34:46 2016

@author: deanpospisil
"""

import sys
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import os

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'net_code/common')
sys.path.append(top_dir + 'net_code/img_gen')
sys.path.append( top_dir + 'xarray/')

import d_curve as dc
import d_misc as dm
import base_shape_gen as bg

saveDir = top_dir + 'net_code/images/baseimgs/'
dm.ifNoDirMakeDir(saveDir)

baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
baseImage = baseImageList[0]

frac_of_image = 0.25
dm.ifNoDirMakeDir(saveDir + baseImage +'/')

if baseImage is baseImageList[0]:

#    os.chdir( saveDir + baseImageList[0])
    mat = l.loadmat(top_dir + 'net_code/img_gen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])
    s = [shape[:-1,:] for shape in s]

elif baseImage is baseImageList[1]:
    nPts = 1000
    s = dc.make_n_natural_formlets(n=20,
                nPts=nPts, radius=1, nFormlets=32, meanFormDir=np.pi,
                stdFormDir=2*np.pi, meanFormDist=1, stdFormDist=0.1,
                startSigma=3, endSigma=0.1, randseed=1, min_n_pix=64,
                frac_image=frac_of_image)
elif baseImage is baseImageList[2]:
    #    os.chdir( saveDir + baseImageList[0])
    mat = l.loadmat(top_dir + 'net_code' + '/img_gen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])
    #adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321]
    a = np.hstack((range(14), range(18,318)))
    a = np.hstack((a, range(322, 370)))
    s = s[a]
    s = [shape[:-1,:] for shape in s]

elif baseImage is baseImageList[3]:
    print('to do')

s = bg.center_boundary(s)
if baseImage is baseImageList[2] or baseImage is baseImageList[0]:
    adjust_c = 4. # cuvature values weren't matching files from Pasu so I scaled them
else:
    adjust_c = 1.

downsamp = 10
shape_dict_list = [{'curvature': -((2. / (1 + np.exp(-0.125 * 
                    dc.curve_curvature(cs) * adjust_c))) - 1)[::downsamp],
                    'orientation': ((np.angle(dc.curveAngularPos(cs))) 
                    % (np.pi * 2))[::downsamp]}
                    for cs in
                    map(lambda shape: shape[:, 1]*1j + shape[:, 0], s)]


import apc_model_fit as ac
maxAngSD = np.deg2rad(171)
minAngSD = np.deg2rad(23)
maxCurSD = 0.98
minCurSD = 0.09
nMeans = 10
nSD = 10
fn = 'apc_test.nc'
import pickle

if 'dmod_new' in locals():
    dmod_new = ac.make_apc_models(shape_dict_list, range(370), fn, nMeans, nSD,
                          maxAngSD, minAngSD, maxCurSD, minCurSD,
                          model_params_dict=None, prov_commit=False, cart=True,
                          save=False)


    with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
        shape_dict_list2 = pickle.load(f)

    dmod_old = ac.make_apc_models(shape_dict_list2, range(370), fn, nMeans, nSD,
                          maxAngSD, minAngSD, maxCurSD, minCurSD,
                          model_params_dict=None, prov_commit=False, cart=True,
                          save=False)


cor = np.dot(dmod_old.values.T, dmod_new.values)
#
#ind = -1
#def key_event(e):
#    global ind
#    ind = ind+1
#    print(ind)
#    plt.subplot(2,1,1)
#    plt.gca().cla()
#    plt.scatter(np.rad2deg(shape_dict_list[ind]['orientation']), shape_dict_list[ind]['curvature'])
#    plt.scatter(np.rad2deg(shape_dict_list2[ind]['orientation']), shape_dict_list2[ind]['curvature'], color='r')
#    plt.subplot(2,1,2)
#    plt.gca().cla()
#    plt.scatter(s[ind][:,0],s[ind][:,1])
#    plt.subplot(2,1,2)
#    plt.show()
#
#fig = plt.figure()
#fig.canvas.mpl_connect('key_press_event', key_event)
#ax = fig.add_subplot(111)
#
#plt.show()
#
#plt.figure()
#plt.imshow(cor)
plt.close('all')
ind = 24
#plt.subplot(4,1,1)
#plt.gca().cla()
#plt.scatter(np.rad2deg(shape_dict_list[ind]['orientation']), shape_dict_list[ind]['curvature'])
#plt.scatter(np.rad2deg(shape_dict_list2[ind]['orientation']), shape_dict_list2[ind]['curvature'], color='r')
#plt.xlabel('orientation')
#plt.ylabel('curvature')
#
#
#plt.subplot(4,1,2)
#plt.gca().cla()
#plt.scatter(s[ind][:,0],s[ind][:,1])
#plt.axis('equal')
#
#
#plt.subplot(4,1,3)
old_new_r = np.array([cor[i,i] for i in range(cor.shape[0])])
plt.plot(dmod_new.coords['cur_mean'], color='cyan')
plt.plot(dmod_new.coords['or_sd'], color='green')
plt.plot(dmod_new.coords['cur_sd'], color='red')
plt.plot(old_new_r, color='blue')
plt.legend(('Mean Curvature', 'ori sd','cur sd', 'r dense vs sparse'))
plt.xlabel('Model Number')
plt.tight_layout()



#m = l.loadmat(top_dir + 'net_code/data/responses/V4_370PC2001.mat')
#
#v4=m['resp'][0][0]
#
#v4_da = xr.DataArray(v4, dims=['unit','shapes']).chunk()
#dmod_old = dmod_old.chunk()
#dmod_new = dmod_new.chunk()
#
#cor_old = ac.cor_resp_to_model(v4_da, dmod_old, fit_over_dims=None, prov_commit=False)
#cor_new = ac.cor_resp_to_model(v4_da, dmod_new, fit_over_dims=None, prov_commit=False)
#cor_con = cor_old-cor_new
#plt.subplot(4,1,4)
#cor_con.plot()
#plt.ylabel('r_sparse - r_dense')
#plt.plot(cor_old.coords['cur_mean']/10)
#np.corrcoef(cor_old.coords['or_mean'], cor_con )

#plt.plot(dmod_new.coords['or_sd'])
#plt.plot(dmod_new.coords['cur_sd'])
#plt.plot(dmod_new.coords['or_mean'])

