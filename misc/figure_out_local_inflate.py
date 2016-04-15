# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:29:20 2016

@author: deanpospisil
"""

import sys, os
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import pickle
import itertools

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'net_code/common')
sys.path.append(top_dir + 'net_code/img_gen')
sys.path.append( top_dir + 'xarray/')

import d_curve as dc
import base_shape_gen as bg

import scipy.signal as si

frac_of_image = 0.25

def cur_or_dict(s,norm=True):
    cs = s[:, 1]*1j + s[:, 0]
    downsamp = 1
    if norm:
        adjust_c = 1 # cuvature values weren't matching files I got so I scaled them
        a = {'curvature': 
        -((2. / (1 + np.exp(-0.125 * dc.curve_curvature(cs)* adjust_c)))-1)[::downsamp],
        'orientation': 
        ((np.angle(dc.curveAngularPos(cs)))% (np.pi * 2))[::downsamp]}
    else:

        a = {'curvature': 
        - dc.curve_curvature(cs)[::downsamp],
        'orientation': 
        ((np.angle(dc.curveAngularPos(cs)))% (np.pi * 2))[::downsamp]}
        
        
    return a
    
def match_ori_max_cur(shape_dict_list_pasu, ws):  
    or_dense = ws['orientation']
    or_pasu = shape_dict_list_pasu[shape_id]['orientation']
    cur_dense = ws['curvature']
    
    or_dif = abs(np.expand_dims(or_pasu, axis=1)-np.expand_dims(or_dense,axis=0))
    min_or_dif = np.pi/20
    close_bool = list(or_dif < min_or_dif)
    close_inds = [np.array(np.where(a_close_bool)).T for a_close_bool in close_bool]
    
#    #select based on closeness to original Pasu, or and cur.
#    match_loc = [close_ind_set[np.argmin(
#                    abs(cur_dense[close_ind_set] - cur_pasu_point))][0].astype(int)
#                    for close_ind_set, cur_pasu_point 
#                    in zip(close_inds, cur_pasu)]
                    
    match_loc = [close_ind_set[np.argmax(abs(cur_dense[close_ind_set]))][0]
                for close_ind_set
                in close_inds]
                
    return match_loc

if 'shape_dict_list_pasu' not in locals():
    
    with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
        shape_dict_list_pasu = pickle.load(f)
        
    mat = l.loadmat(top_dir + 'net_code/img_gen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])
    s = [shape[:-1,:] for shape in s]
    s = bg.center_boundary(s)

normed = True
shape_id = 172
rect_len = 8

shape_dict_list_dense = (cur_or_dict(ashape / np.max(np.abs(ashape)), norm=normed)
                         for ashape in s)

ws = itertools.islice(shape_dict_list_dense, shape_id, shape_id+1).__next__()
dense_val = np.array([ws['curvature'], 
                      ws['orientation']]).T

orig_val = np.array([shape_dict_list_pasu[shape_id]['curvature'], 
                     shape_dict_list_pasu[shape_id]['orientation']]).T


match_loc_orig = match_ori_max_cur(shape_dict_list_pasu, ws)

ashape = s[shape_id]

deflate = np.fft.fftshift(np.expand_dims((1/((si.gaussian(800,30)*0.3)+1)), axis=1))
shift = match_loc_orig[1]-60
deflate  = deflate [[(x + shift)%len(deflate) for x in range(len(deflate))]]
l_ashape =  deflate * ashape

l#_ashape = l_ashape/np.max(np.abs(l_ashape))
apc_dict = cur_or_dict(l_ashape, norm=normed)
soft_val = np.array([ apc_dict[key] for key in [ 'curvature','orientation']]).T

match_loc_soft = match_ori_max_cur(shape_dict_list_pasu, apc_dict)



plt.close('all')
plt.subplot(221)
plt.plot(dense_val[:, 1], dense_val[:, 0])         
plt.scatter(dense_val[match_loc_orig, 1], dense_val[match_loc_orig, 0], color='r')         
plt.scatter(orig_val[:,1], orig_val[:,0], color='r')

plt.subplot(222)
plt.plot(ashape[:, 1], ashape[:, 0])
plt.scatter(ashape[match_loc_orig, 1], ashape[match_loc_orig, 0], color='r')
plt.axis('equal')

plt.subplot(223)
plt.plot(soft_val[:,1], soft_val[:,0])         
plt.scatter(soft_val[match_loc_soft, 1], soft_val[match_loc_soft, 0], color='g')         
plt.scatter(dense_val[match_loc_orig, 1], dense_val[match_loc_orig, 0], color='r')

plt.subplot(224)
plt.plot(l_ashape[:,1], l_ashape[:,0])
plt.scatter(l_ashape[match_loc_soft, 1], l_ashape[match_loc_soft, 0], color='g')
plt.axis('equal')









'''
ind = -1
def key_event(e):

    global ind
    ind = ind+1
    print(ind)
    plt.gca().cla()
    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation']), shape_dict_list[ind]['curvature'])
    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation'][0]), shape_dict_list[ind]['curvature'][0], color='g')
    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation'][20]), shape_dict_list[ind]['curvature'][20], color='y')

    plt.scatter( np.rad2deg(shape_dict_list2[ind]['orientation']), shape_dict_list2[ind]['curvature'], color='r')
    plt.show()

#
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)

plt.show()
'''