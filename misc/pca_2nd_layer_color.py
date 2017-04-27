# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:48:42 2016

@author: deanpospisil
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:45:44 2016

@author: deanpospisil
"""
import matplotlib.colors as clrs


import sys, os
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import pickle

top_dir = os.getcwd().split('net_code')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'net_code/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
sys.path.append(top_dir + 'nets')
plt.close('all')

import d_misc as dm
import d_img_process as imp
import pandas as pd
from sklearn import  decomposition

def my_cor(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    r = np.dot(a, b)
    return r
def vis_square(data, padsize=0, padval=0):

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data, interpolation='nearest', cmap = cm.hot, vmin=0, vmax=1)
    plt.colorbar()

    plt.tight_layout()
    return data

def vis_square_rgb(data, padsize=0, padval=0):

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data, interpolation='nearest')

    plt.tight_layout()
    return data


def get2dCfIndex(xsamps, ysamps,fs):
    fx, fy = np.meshgrid(np.fft.fftfreq(int(xsamps),1./fs),
                         np.fft.fftfreq(int(ysamps),1./fs) )
    c = fx + 1j * fy
    return c

if 'afile' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            afile = pickle.load(f, encoding='latin1')
        except:
            afile = pickle.load(f)


layer = 0
sample_rate_mult = 10
ims = afile[layer][1]

ims = np.array([im for im in ims])
ims = np.sum(ims, 1)[:48, ...]
ims = ims - np.mean(ims,axis =(1,2), keepdims=True)
fims = np.abs(np.fft.fft2(ims, s=np.array(np.shape(ims)[1:])*sample_rate_mult))
fims_or_index = np.max(fims, axis=(1,2))/np.sum(fims, axis=(1,2))
oriented_index = fims_or_index>np.percentile(fims_or_index, 20)
fims = fims[oriented_index, ...]

fims = fims.reshape(np.shape(fims)[0], (11*sample_rate_mult)**2)
c = get2dCfIndex(11*sample_rate_mult, 11*sample_rate_mult, 11*sample_rate_mult)
mag = np.abs(c).ravel()
ang = np.angle(c)
ang = ang.ravel()
fims = fims[:, ang>0]
ang = ang[ang>0]
ors = (ang[np.argmax(fims, axis=1)] - np.pi/2) % np.pi

ors = np.squeeze(ors)
sorsi = np.argsort(ors)
ors = ors[sorsi]

ims_2 = afile[layer+1][1][:128, oriented_index,...]
#ims_2 = ims_2.reshape( np.shape(ims_2)[:-2] + (np.shape(ims_2)[-1]**2,))
#ims_2 = ims_2[:,sorsi,:]

#ims_2 = ims_2.swapaxes(1,2)

#plt.plot(np.sqrt(var_explained[:,0].T))
def unrav_over_last_n(a, n=2):
    return a.reshape(np.shape(a)[:-n] + (np.product(np.shape(a)[-n:]),))
    
ims_2_unrav=unrav_over_last_n(ims_2)
u, s, v = np.linalg.svd(ims_2_unrav, full_matrices=False)
s_var = s**2
var_explained = np.cumsum(s_var / np.sum(s_var, 1, keepdims=True), axis=1)


plt.figure()
plt.plot(np.arange(5)+1, var_explained[:,:5].T)
plt.title('2nd Layer Filter Weights Fraction Variance Explained by PCA')
plt.xlabel('Principal Components', fontsize='large')
plt.ylabel('Fraction Variance Explained', fontsize='large')
plt.ylim(0,1)
plt.xlim(1,5)
plt.xticks(list(np.arange(5)+1))
plt.gca().xaxis.set_ticks_position('none') 
plt.gca().yaxis.set_ticks_position('none') 

import husl
def cart2angle(a):
    ang = np.array([np.arctan2(cart[0, :], cart[1, :]) for cart in a ])
    ang = np.rad2deg(ang)%360
    return ang
def cart2mag(a):
    mag = np.sum(a**2, 1)**0.5
    return mag
def cart2pol(a):
    angle = cart2angle(a)
    mag = cart2mag(a)
    pol = np.dstack((angle, mag)).swapaxes(1,2)
    return pol
def ziphusl(a):
    rgb = husl.huslp_to_rgb(a[0], a[1], a[2])
    return rgb
   
freq=2
sat_scale = 100
cor_scale = 80

pred = np.array([np.cos(freq*ors), np.sin(freq*ors)]).T
pred = None
kernels = ims_2

#def lsq_coeff_image(kernels, pred=None):
cor_is_lum = True
kernels = unrav_over_last_n(kernels)    
if pred is None:
    u, s, v = np.linalg.svd(kernels, full_matrices=False)
    pred = u[..., :2]
    #make pred the 1st two princomps
else:
    if pred.shape[1]!=2 and pred.shape[2]==2:
        pred = pred.T
    pred = pred / np.linalg.norm(pred, 1, keepdims=True)
    pred = np.tile(pred, (kernels.shape[0], 1, 1))

coeffs, res, _, _ = map(np.array, zip(*[np.linalg.lstsq(a_pred, a_kern) 
                                   for a_pred, a_kern 
                                   in zip(pred, kernels)]))
coeffs_pol = cart2pol(coeffs) 
if cor_is_lum:
    cor = (1-res/(np.sum(kernels**2, 1)))**0.5
    cor = np.expand_dims(cor,1)*cor_scale
    coeffs_pol = np.concatenate((coeffs_pol[:,[0,],:], cor),1)

coeffs_pol_hsl = np.concatenate((coeffs_pol, sat_scale * np.ones_like(coeffs_pol[:, [0], :])), 1)
coeffs_pol_hsl = coeffs_pol_hsl[:,[0, 2, 1], ...]#by default is hue:angle, sat:mag, lum:constant
coeffs_pol_rgb = np.apply_along_axis(ziphusl, 1, coeffs_pol_hsl)
coeffs_pol_rgb_img = np.reshape(coeffs_pol_rgb, (128, 3, 5, 5)).swapaxes(-1, 1)

plt.figure()
plt.subplot(133)
vis_square_rgb(coeffs_pol_rgb_img, padsize=1, padval=0)
plt.xticks([])
plt.yticks([])

plt.subplot(121)
nx, ny = (100, 100)
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
xv, yv = np.meshgrid(x, y)
cart = xv*1j + yv
pol = (np.rad2deg(np.angle(cart)))%360
mag = abs(cart)
rgb = np.apply_along_axis(ziphusl, 2, np.dstack((pol, sat_scale*np.ones_like(mag), cor_scale*mag)))
plt.imshow(rgb, interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#now plot on top of it.
unit = -6 #plot the sinusoidal unit with its weights organized by orientation
unit = 122 #plot the sinusoidal unit with its weights organized by orientation

scale = 100/np.max(np.sum(coeffs[unit, :, :]**2)**0.5)
shift = 50 
plt.scatter((scale*np.squeeze(coeffs[unit, 0, :]))+shift, 
                            (scale*np.squeeze(coeffs[unit, 1, :]))+shift,
                            color=coeffs_pol_rgb[unit, ...].swapaxes(0, 1), edgecolors='black')


plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(coeffs_pol_rgb_img[unit, ...], interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.title('Kernel: '+str(unit))

plt.tight_layout()
#
#rgb = np.array([husl.huslp_to_rgb(hsv_sec[0], hsv_sec[1], hsv_sec[2]) 
#                for hsv_sec in x_pol_sc_unrv.T]).T
#rgb = rgb.reshape(x_pol_sc.shape)
#    

        

    

##
#inds = np.argsort(var_explained[:,1])
#ind = 33
##pc = np.dot(np.squeeze(u[inds[ind],:,:2]).T, ims_2[inds[ind],:,:])
##plt.scatter(pc[0,:], pc[1,:])
##plt.scatter(ors, u[inds[ind],:,0])
#plt.scatter(ors, u[ind,:,1])
#
#from skimage import io, color


