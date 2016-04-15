# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:06:19 2016

@author: dean
"""
import numpy as np
import matplotlib.pyplot as plt


def polar_rect_func(f_ind, up_r, down_r, up_ang, down_ang):
    im_ang = np.rad2deg(np.angle(f_ind)) + 180.
    im_bool_ang = ((im_ang>down_ang) * (im_ang<up_ang))
    im_bool_r = ((np.abs(np.real(f_ind))>down_r) + (np.abs(np.imag(f_ind))>down_r)) * \
            ((np.abs(np.real(f_ind))<up_r) * (np.abs(np.imag(f_ind))<up_r))

    p_rect = im_bool_r * im_bool_ang
    return p_rect.astype(np.double)


ang_list = [np.array([[   0. ,   22.5,   45. ,   67.5,   90. ,  112.5,  135. ,  157.5],
                      [  22.5,   45. ,   67.5,   90. ,  112.5,  135. ,  157.5,  180. ]])]
theta_dif = ang_list[0][0,1] - ang_list[0][0,0]

f_cuts_list = [np.array([ls, ls*2])]

n_scale = 3
ls = 0.01
r_dif = ls

for scale in range(1, n_scale):
    low_ang = np.hstack((ang_list[scale-1][0, :] + theta_dif/(2**scale), ang_list[scale-1][0, :] ))
    up_ang = low_ang + theta_dif/(2**scale)
    ang_list.append(np.vstack((np.sort(low_ang), np.sort(up_ang))))
    f_cuts_list.append(np.array([f_cuts_list[scale-1][1], f_cuts_list[scale-1][1] + (2**scale)*r_dif]))


imdim = 1024
im = np.zeros((imdim,imdim))
f_ind = np.expand_dims(np.fft.fftfreq(imdim), axis=1)
f_ind = f_ind.T + np.flipud(f_ind*1j)
im_ang = np.rad2deg(np.angle(f_ind)) + 180.
im_r = np.abs(f_ind)


imf = []
for scale, r_band in enumerate(f_cuts_list):
    for ang_band in ang_list[scale].T:
        imf.append(polar_rect_func(f_ind, r_band[1], r_band[0], ang_band[1], ang_band[0]))

ind = 30
plt.subplot(2,1,1)
plt.imshow(np.fft.fftshift(imf[ind]))
plt.subplot(2,1,2)
plt.imshow(np.fft.fftshift(np.imag(np.fft.ifft2(imf[ind]))),interpolation='none')
for ang in ang_list[0].T:
    print(ang)