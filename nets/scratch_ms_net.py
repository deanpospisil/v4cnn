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

n_bin_edges = 10
#fewest_freqs = 10.
#highest_divisor = np.floor(nyq / fewest_freqs)
#bin_edges = nyq/np.logspace(1, np.log2(highest_divisor), num=n_bin_edges, base=2)
#need to work on hwo to do spacing a little bit
bin_upper_edges = nyq/np.geomspace(1,20, num=n_bin_edges)
bin_edges = np.append(bin_upper_edges, 0)
bin_half_width = np.diff(bin_edges)/2
bin_centers = [(bin_edges[i+1] + bin_edges[i]) / 2. for i in range(len(bin_edges))[:-1]]
for i in range(len(bin_centers)):
    filt = myGuassian(mag, bin_centers[i], bin_half_width[i])
    plt.figure()
    plt.subplot(121)
    plt.imshow(filt)
    plt.subplot(122)
    plt.imshow(np.fft.fftshift(np.fft.irfft2(filt)))
    
    
    
    