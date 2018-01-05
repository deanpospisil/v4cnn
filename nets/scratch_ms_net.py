# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:12:44 2017

@author: dean
"""
import numpy as np
import matplotlib.pyplot as plt
def getfIndex(nSamps, fs):

    f = np.fft.fftfreq(nSamps, 1./fs)
#    nSamps=np.double(nSamps)
#    fs=np.double(fs)
#    nyq = fs/2
#    df = fs / nSamps
#    f = np.arange(nSamps) * df
#    f[f>nyq] = f[f>nyq] - nyq*2
    return f

#
kernel_widths = [3, 2, 2]
strides = [2, 2, 1]

kernel_widths = [3,]
strides = [2,]
def rf_width(kernel_widths, strides):
    rf_width = [1,]
    strides = np.array(strides)
    kernel_widths = np.array(kernel_widths)
    
    kernel_widths = np.insert(kernel_widths, 0, 1)
    strides = np.insert(strides, 0, 1)
    
    stride_prod = np.cumprod(strides)
    
    for i in range(len(kernel_widths))[1:]:
        rf_width.append(rf_width[i-1] + (kernel_widths[i] - 1)*stride_prod[i-1])
    
    return rf_width[1:]

def output_sizes(kernel_widths, strides, input_size):
    if not (type(input_size) is type(list())):
        input_size = [input_size,] 
        
    for i in range(len(strides)):
        input_size.append(np.ceil((input_size[i] - kernel_widths[i]) 
                         / strides[i] + 1))

def centered_cut(o_length, n_length):

    if o_length % 2: #if the image has odd number rows
        if not n_length % 2: #if the  new length is even
            n_length = n_length + 1 #make it odd
        center = (o_length - 1)/2 
        half_width = (n_length - 1)/2
        r1 = center - half_width
        r2 = center + half_width + 1
    else:
        if n_length % 2: #if the new length is odd
            n_length = n_length + 1 #make it even
        center = (o_length)/2
        half_width = n_length/2
        r1 = center - half_width
        r2 = center + half_width 
    return (int(r1), int(r2))

def centered_crop(img, r, c):
    n_r_img = len(img[:, 0])
    n_c_img = len(img[0, :])
    (r1, r2) = centered_cut(n_r_img, r)
    (c1, c2) = centered_cut(n_c_img, c)
           
    return img[r1:r2, c1:c2]

n = 227
img = np.ones((n, n))

row_freq = np.fft.fftfreq(n, 1./n)
col_freq = np.fft.rfftfreq(n, 1./n)

nyq = np.max(row_freq) 

row_freq = np.broadcast_to(np.expand_dims(row_freq,1), 
                           (len(row_freq), len(col_freq)))
col_freq = np.broadcast_to(np.expand_dims(col_freq, 0), 
                           (len(row_freq), len(col_freq)))
mag = (row_freq**2 + col_freq**2)**0.5

def myGuassian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

filt = myGuassian(mag, 25, 10)

#plt.imshow(filt)
from scipy import misc
im = face = misc.face()
im = im[:n,:n, :]/255.
im = np.mean(im, -1)
im= 1-im
plt.imshow(im, cmap='Greys')
n_bin_edges = 10
#fewest_freqs = 10.
#highest_divisor = np.floor(nyq / fewest_freqs)
#bin_edges = nyq/np.logspace(1, np.log2(highest_divisor), num=n_bin_edges, base=2)
#need to work on hwo to do spacing a little bit
bin_upper_edges = nyq/np.geomspace(1, n/10., num=n_bin_edges)
bin_edges = np.append(bin_upper_edges, 0)
bin_half_width = np.abs(np.diff(bin_edges)/2)
bin_centers = [(bin_edges[i+1] + bin_edges[i]) / 2. for i in range(len(bin_edges))[:-1]]
nyq = np.ceil((n/2.)/(bin_centers+bin_half_width*2))

for i in range(len(bin_centers))[:]:
    filt = myGuassian(mag, bin_centers[i], bin_half_width[i])
    plt.figure()
    plt.subplot(141)
    plt.imshow(filt)
    plt.xticks([]);plt.yticks([])
    plt.subplot(142)
    plt.imshow(np.fft.fftshift(np.fft.irfft2(filt)), cmap='Greys')
    plt.xticks([]);plt.yticks([])
    plt.subplot(143)
    filt_im = np.fft.irfft2(filt * np.fft.rfft2(im))
    plt.imshow(filt_im, cmap='Greys', interpolation='nearest');plt.xticks([]);plt.yticks([])
    plt.subplot(143)
    filt_im = np.fft.irfft2(filt * np.fft.rfft2(im))
    plt.imshow(filt_im[::int(nyq[i]), ::int(nyq[i])], cmap='Greys', interpolation='nearest');plt.xticks([]);plt.yticks([])
    


spatial_sd = 3
cropped_t_filt = []
for i in range(len(bin_centers))[:]:
    sigma = bin_half_width[i]
    mu = bin_centers[i]
    
    one_sd = 1./(sigma * (2*np.pi/n))
    spatial_extent = int((one_sd*2)*spatial_sd)

    
    f_filt = myGuassian(mag, mu, sigma) 
    t_filt = np.fft.fftshift(np.fft.irfft2(f_filt, (n, n)))
    cropped_t_filt.append(centered_crop(t_filt, spatial_extent, spatial_extent))     


#%% calculating potential RF sizes. 
r = np.linspace(1,100,1000)
n = 1
m = 200  
sigma_f = ((np.sqrt((-2.*m*r - np.pi*n*r + 2.*np.pi*r)**2. + 16.*np.pi*m*r**2.)
         + 2.*m*r + np.pi*n*r - 2.*np.pi*r)/(4.*np.pi*r**2.))
t = 1./((sigma_f + r**-1)*2)

rf = (1./((1./r + sigma_f)*2)) * n + 2.*m/(sigma_f*2.*np.pi)
plt.figure()
plt.plot(rf, sigma_f);plt.xlabel('rf');plt.ylabel('sigma bw')
plt.figure()
plt.plot(rf, t);plt.xlabel('rf');plt.ylabel('sampling period')
plt.figure()
plt.plot(rf, 1./rf);plt.xlabel('rf');plt.ylabel('freq')
plt.figure()
plt.plot(rf, t);plt.xlabel('rf');plt.ylabel('sampling period')

##spatial filts
#spatial_sd = 3
#for i in range(len(bin_centers))[:-1]:
#    sigma = bin_half_width[i]
#    mu = bin_centers[i]
#    f_filt = myGuassian(mag, mu, sigma)
#    one_sd = 1./(sigma*(2*np.pi/n))
#    spatial_extent = int((one_sd*2)*spatial_sd)
#    
#    t_filt = np.fft.irfft2(f_filt, (n, n))
#    shift_t_filt = np.fft.fftshift(t_filt)
#
#    cropped_t_filt = centered_crop(shift_t_filt, spatial_extent, spatial_extent)
#    plt.figure()
#    plt.imshow(cropped_t_filt)

#%%
import os
import sys
import re
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0] 
sys.path.append(top_dir+ 'v4cnn')
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')
sys.path.append( top_dir + 'nets/')

import caffe_net_response as cf
import d_misc as dm
import xarray as xr
import apc_model_fit as ac
import d_curve as dc
import d_img_process as imp
import scipy.io as l
import caffe
from caffe import layers as L
from caffe import params as P
n = caffe.NetSpec()
n.data, n.label = L.Data(batch_size=256, transform_param=dict(mirror=True, crop_size=227),
                                ntop=2)

L.Convolution(n.data, ntop=1, )
print(n.to_proto())
#%%
def example_network(batch_size):


    n.loss, n.label = L.Data(shape=[dict(dim=[1]),
                                         dict(dim=[1])],
                                  transform_param=dict(scale=1.0/255.0),
                                  ntop=2)

    n.accuracy = L.Python(n.loss, n.label,
                          python_param=dict(
                                          module='python_accuracy',
                                          layer='PythonAccuracy',
                                          param_str='{ "param_name": param_value }'),
                          ntop=1,)

    return n.to_proto()
a = example_network(20)
