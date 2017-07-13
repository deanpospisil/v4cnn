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

n_bin_edges = 6
#fewest_freqs = 10.
#highest_divisor = np.floor(nyq / fewest_freqs)
#bin_edges = nyq/np.logspace(1, np.log2(highest_divisor), num=n_bin_edges, base=2)
#need to work on hwo to do spacing a little bit
bin_upper_edges = nyq/np.geomspace(1, n/10., num=n_bin_edges)
bin_edges = np.append(bin_upper_edges, 0)
bin_half_width = np.abs(np.diff(bin_edges)/2)
bin_centers = [(bin_edges[i+1] + bin_edges[i]) / 2. for i in range(len(bin_edges))[:-1]]
for i in range(len(bin_centers)):
    filt = myGuassian(mag, bin_centers[i], bin_half_width[i])
    plt.figure()
    plt.subplot(121)
    plt.imshow(filt)
    plt.subplot(122)
    plt.imshow(np.fft.fftshift(np.fft.irfft2(filt)))

def centeredCrop(img, new_height, new_width):
   width =  np.size(img,1)
   height =  np.size(img,0)

   #if an odd number defaults to putting extra pixel to left and top
   left = int(np.ceil((width - new_width)/2.))
   top = int(np.ceil((height - new_height)/2.))
   right = int(np.floor((width + new_width)/2.))
   bottom = int(np.floor((height + new_height)/2.))

   cImg = img[top:bottom, left:right]
   return cImg
#spatial filts
for i in range(len(bin_centers))[:-1]:
    f_filt = myGuassian(mag, bin_centers[i], bin_half_width[i])
    t_filt = np.fft.irfft2(f_filt)
    shift_t_filt = np.fft.fftshift(t_filt)
    shift_t_filt_pwr = np.sum(shift_t_filt**2, 1)
    cut_off = np.max(shift_t_filt_pwr)/10.
    cut_off_pix = [i for i in range(len(shift_t_filt_pwr)) if (shift_t_filt_pwr[i])>cut_off][0]
    cut_off_pix = int(len(shift_t_filt_pwr)/2. - cut_off_pix)
    cropped_t_filt = centeredCrop(shift_t_filt, cut_off_pix*5, cut_off_pix*5)
    plt.figure()
    plt.subplot(121)
    plt.imshow(cropped_t_filt)
#%%
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
