# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:19:44 2017

@author: deanpospisil
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sc

top_dir = os.getcwd().split('v4cnn')[0]
save_dir = top_dir + 'v4cnn/images/baseimgs/sinusoids/'

ds = 1
imshape = (20, 20)
rft = np.fft.rfft2(np.ones(imshape))
rft_shape = rft.shape
ims = []
count = -1
for r in range(rft_shape[0])[::ds]:
    for c in range(rft_shape[1])[::ds]:
        count = count + 1
        print(r)
        print(c)
        rft = np.zeros(imshape)
        rft[r, c] = 1
        im = np.real(np.fft.ifft2(rft, imshape))
        im = im - im.min()
        im = im*255.
        ims.append(im)
        np.save(save_dir + str(count), im)
        sc.misc.imsave(save_dir + str(count) + '.bmp', im)
        
r = np.fft.fftfreq(imshape[0]).reshape((imshape[0]), 1)
c = np.fft.fftfreq(imshape[1]).reshape((1, imshape[1]))

cart_c = c + r*1j
freq_index = cart_c[:rft_shape[0], :rft_shape[1]].ravel()

plt.imshow(np.abs(cart_c))
#plt.imshow(np.angle(cart_c))