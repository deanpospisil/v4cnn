# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:05:38 2016

@author: deanpospisil
"""
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import caffe_net_response as cf
import matplotlib
from matplotlib.ticker import FuncFormatter
import pickle as pk
import xarray as xr;import pandas as pd
import apc_model_fit as ac
import matplotlib.ticker as mtick;
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp


no_rotation = [0, 1, 2, 10, 14, 18, 26,30, 38, 46, 54, 62, 70, 78, 86, 94,102, 110, 118, 126, 134,
 142, 150, 158,166, 174, 182, 190,198, 206, 214, 222, 224, 232, 236, 244, 252, 254, 
 262, 270, 278, 286, 294, 302, 310, 314, 322, 330, 338, 346, 354, ]

no_rotation = [0, 1, 2, 10, 18, 22,30,34,42, 50, 58, 66, 74, 82,  90, 98, 106, 114, 122, 130, 138,
 146, 154, 162,170, 178, 186, 194, 202, 210, 218, 226, 228, 236, 240, 248, 256, 
 258, 266, 274, 282, 290, 298, 306, 314, 322, 330, 338, 346, 354, 362,370 ]
 
rot_shape_resp_list = []
rot_shape_id_list = []
#print(rot_shape_id_list)
v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('unit','shapes')
v4_resp_apc = v4_resp_apc[0]

for i, s_ind in enumerate(no_rotation):
    if i < len(no_rotation)-1:
        last_rot = no_rotation[i+1]
    else:
        last_rot = 370
    #rot_shape_resp_list.append(v4_resp_apc[s_ind:last_rot, 0].values)
    rot_shape_id_list.append(np.arange(s_ind, last_rot))

rot_shape_id_list[3] = rot_shape_id_list[3][:4]
rot_shape_id_list[-7] = rot_shape_id_list[-7][:4]


max_pix_width = [64.,]
img_n_pix = 64
s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(-1, 370); center_image = round(img_n_pix/2)
x = (center_image, center_image, 1);y = (center_image, center_image, 1)
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids, scale=scale, x=x, y=y)
trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict, base_stack, npixels=img_n_pix))
#plot smallest and largest shape
no_blank_image = trans_img_stack[1:]
_ = no_blank_image.copy()
#_[no_blank_image==0] = 255
#_[no_blank_image>0] = 0
no_blank_image = _
_ -= v4_resp_apc.mean()
v4_resp_n = v4_resp_apc/v4_resp_apc.max()


cmap = cm.cool
color_scale = cmap(v4_resp_n)

m = 8
n = 51
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(51.,9.))
plot_id = np.arange(0, m*n).reshape(m, n)
gs_kw = {'nrows':m+2, 'ncols':n, 'width_ratios':[1,]*n, 
         'height_ratios':[1,]*m+[0.5, 0.5]}
gs = gridspec.GridSpec(nrows=m+2, ncols=n, width_ratios=[1,]*n,
                        height_ratios=[1,]*m+[0.5, 0.5])
cbar_ax = plt.subplot(gs[-1, 15:-15])

m_ind = 0
n_ind = 0
shape_tot_ind = 0
for shape_id_set in rot_shape_id_list:
    for shape_id in shape_id_set:
        img = no_blank_image[shape_id]
        color = color_scale[shape_tot_ind]
        color = np.array([0,0,0,1])
        c_img = img.reshape(img.shape+(1,))*color.reshape((1,1,4))
        c_img /= c_img.max()
        p_num = plot_id[m_ind, n_ind]
        ax = plt.subplot(gs[p_num])
        ax.imshow(c_img, interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([]);
        ax.set_xlim([0,64])
        ax.set_ylim([0,64])
        m_ind+=1
        shape_tot_ind+=1
        
        for spine in ['bottom','left','right','top']:
            ax.spines[spine].set_visible(False)
    m_ind=0
    n_ind+=1 
    

import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=1)

cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal',ticks=[0, 1],)
cb1.set_label('Normalized Response', fontsize=80, labelpad=0)
cbar_ax.set_xticks([0, 1])
cbar_ax.set_xticklabels(['0','1'], fontsize=80)

ax.figure.tight_layout(w_pad=-0.1, h_pad=-.01)

ax.figure.savefig(top_dir + '/analysis/figures/images/apc_fig_black.pdf')
