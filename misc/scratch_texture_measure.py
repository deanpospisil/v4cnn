# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:51:19 2018

@author: deanpospisil
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import convolve2d as conv
import scipy as sc
import os
from skimage.filters import gabor_kernel
#from skimage.filters import gabor
from scipy.misc import imread
import matplotlib.cm as cm

imname = '/Users/deanpospisil/Desktop/modules/v4cnn/images/han.png'

im = imread(imname).sum(-1)
im = im[:149, :149]
plt.imshow(im)

thetas = np.linspace(0, np.pi-np.pi/20., 4)
#frequency is in terms of pixels so 0.1 is a period of 10 pixels
#std = 3 cycles so will be 30 pixels or so large filter.
freqs = np.linspace(0.1,0.5,5)
all_kerns = []
for frequency in freqs:
    kernels = []
    
    temp = [np.real(gabor_kernel(frequency, theta=theta)) for theta in thetas]
    temp = [a_kernel/(np.sum(a_kernel**2)**0.5) for a_kernel in temp]
    
    kernels.append(temp)
    
    temp = [np.imag(gabor_kernel(frequency, theta=theta)) for theta in thetas]
    temp = [a_kernel/(np.sum(a_kernel**2)**0.5) for a_kernel in temp]
    
    kernels.append(temp)
    
    all_kerns.append(kernels)
    
# freq, phase, ori
plt.figure()
plt.imshow(all_kerns[0][1][1])


#%%
res = np.zeros((len(freqs), 2, len(thetas), im.shape[0], im.shape[1]))
for i, freq in enumerate(all_kerns):
    for j, phase in enumerate(freq):
        for k, ori in enumerate(phase):
            res[i,j,k, :, :] = conv(im, ori, mode='same')

simp_res = res[..., 30:-30, 30:-30]
comp_res = (simp_res**2).sum(1)  

   
#%%
#auto = sc.signal.fftconvolve(comp_res,comp_res, mode='same');
#plt.imshow(auto[4,1,...]);plt.colorbar();
#%%
p1 = 0
f1 = 3
p2 = 0
f2 = 1
auto = sc.signal.correlate2d(comp_res[f1,p1,...],comp_res[f2,p2,...], mode='same')


plt.subplot(311)
plt.imshow(comp_res[f1,p1])
plt.xticks([]);plt.yticks([]);

plt.subplot(312)
plt.imshow(comp_res[f2,p2])
plt.xticks([]);plt.yticks([]);

plt.subplot(313)
plt.imshow(im[30:-30, 30:-30])
plt.xticks([]);plt.yticks([]);
plt.figure()
plt.imshow(auto[40:-40,40:-40]);plt.colorbar();

#now if filter size + neighborhood width is the scale of the measurement.


#%%%
Nsc = 4;
Nor = 4;
fim = np.fft.fft2(im)
f_c = np.fft.fftfreq(np.shape(fim)[0])[:, np.newaxis]*np.pi*2;
f_r = f_c.T*1j
f_ind = f_r + f_c;

r_h_wind = np.zeros(np.shape(fim));
r_h_wind = np.abs(f_ind);

r_wind[np.abs(f_ind)>=np.pi/2.] = 1;












