# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:20:36 2017

@author: deanpospisil
"""
import os
import sys

#make the working directory two above this one
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'/common')
sys.path.append(top_dir +'/nets')

import d_img_process as imp
import d_misc as dm
import d_curve as dc
import scipy.io as  l
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from skimage import measure
from scipy import ndimage
from scipy.interpolate import interp1d


im = misc.imread('/Users/deanpospisil/Desktop/modules/v4cnn/img_gen/dp_ang_pos.jpg')
im = im[:,:,1]
# Remove small white regions
#im = misc.imresize(im, size=10.)

plt.hist(im.ravel())

binary_img = im < 200
plt.figure()
plt.imshow(binary_img)
open_img = ndimage.binary_opening(binary_img)
# Remove small black hole
close_img = ndimage.binary_closing(open_img)
plt.figure(figsize=(10,10))
plt.imshow(close_img)
plt.contour(close_img, [0.5], linewidths=2, colors='r')
contours = measure.find_contours(close_img, 0.5)
contour_len = [len(contour[:,0]) for contour in contours]

shapes = [contour for contour in contours if 
          (len(contour[:,0])>100)]
#shapes = [shapes[i] for i in [2,20,31,40,0,19,32,42]]
#plt.figure()
#plt.hist(contour_len,  bins=100, range=(0,5000))
#%%
plt.figure(figsize=(10,10))
for contour in shapes[:]:
    #plt.figure()
    cy = contour[:, 0]
    cx = contour[:, 1]
    plt.plot(cx, -cy)
    plt.axis('equal')

max_boundary = [np.max(np.sqrt(shape[:,0]**2 + shape[:,1]**2)) for shape in shapes]
base_stack = dc.center_boundary(np.array(shapes))/max_boundary
plt.figure(figsize=(10,10))

base_stack = list(base_stack[:6]) + [base_stack[-1],] + list(base_stack[6:-1])

for shape in base_stack:
    plt.figure()
    plt.plot(shape[:, 0], shape[:, 1])
    plt.axis('square')
    plt.xlim(-1.5,1.5);plt.ylim(-1.5,1.5)
    plt.xticks([]);plt.yticks([])

#swap1 = base_stack[1]
#swap2 = base_stack[2]
#base_stack[1] = swap2
#base_stack[2] = swap1



np.save(top_dir + 'img_gen/dp_ang_pos_verts', base_stack)
bs2=np.load(top_dir + 'img_gen/dp_ang_pos_verts.npy')


#%%
mg_n_pix = 227
max_pix_width = [64.,]

s = np.load(top_dir + 'img_gen/dp_ang_pos_verts.npy')


base_stack = imp.center_boundary(s)
ang_pos = []

for shape_ind in range(9):
    x = base_stack[shape_ind][:,0][::40]
    y = base_stack[shape_ind][:,1][::40]
    x[-1] = x[0]
    y[-1] = y[0]
    
    dom = np.arange(len(x))
    fx = interp1d(dom, x, kind=3)
    fy = interp1d(dom, y, kind=3)
    
    newdom = np.linspace(0, len(x)-1, 1000)
    newx = fx(newdom)
    newy = fy(newdom)
    c_shape = newx + newy*1j
    
    
    angle = dc.curveAngularPos(c_shape)
    curve = dc.curve_curvature(c_shape)
    hi_curv_ind = np.argmax(np.abs(curve[2:]))
    ang_pos_hicurv = angle[hi_curv_ind]
    cx, cy = dc.get_center_boundary(np.real(c_shape), np.imag(c_shape))
    
    #plt.plot(-curve[2:])
    plt.figure(figsize=(5,5))
    plt.plot(newx, newy)
    plt.scatter(cx,cy)
    plt.scatter(x, y)
    plt.axis('square')
    plt.title(np.round(np.rad2deg(np.angle(ang_pos_hicurv))))
    plt.scatter(newx[hi_curv_ind], newy[hi_curv_ind])
    plt.plot(x,y)
    ang_pos.append(np.angle(ang_pos_hicurv))
    
#%%
def cart2polar(x, y):
    rho = (x**2 + y**2)**0.5
    theta = np.arctan2(x, y)
    return rho, theta
def pol2cart(rho, theta):
    y = rho * np.cos(theta)
    x = rho * np.sin(theta)
    return x, y

n_plots = len(ang_pos)
plt.figure(figsize=(2,10))
shift = []
non_shift = []
for shape_ind, the_ang_pos in enumerate(ang_pos):
    x = base_stack[shape_ind][:,0]
    y = base_stack[shape_ind][:,1]
    non_shift.append(np.array([x, y]).T)
    
    rho, theta = cart2polar(x, y)
    rho = rho/np.max(rho)
    xnew, ynew = pol2cart(rho, theta + the_ang_pos + np.pi)
    shift.append(np.array([xnew, ynew]).T)
    
    plt.subplot(n_plots, 1, shape_ind+1)
    plt.plot(x,y)
    plt.plot(xnew,ynew, color='r')
    plt.axis('square')
    plt.xlim(-1.5,1.5);plt.ylim(-1.5,1.5)
    plt.xticks([]);plt.yticks([])

both_shift = non_shift + shift
np.save(top_dir + 'img_gen/dp_ang_pos_verts_shift', both_shift)
bs2=np.load(top_dir + 'img_gen/dp_ang_pos_verts_shift.npy')

