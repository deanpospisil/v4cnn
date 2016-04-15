# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 20:05:48 2016

@author: deanpospisil
"""

#figure out curvelet
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

imdim = 512
im = np.zeros((imdim,imdim))
f_ind = np.expand_dims(np.fft.fftfreq(imdim), axis=1)
f_ind = f_ind.T + np.flipud(f_ind*1j)
flat_f_ind = np.expand_dims(f_ind.flatten(), axis=0)
r = np.abs(flat_f_ind)
theta = np.angle(flat_f_ind)
#plt.imshow(np.fft.fftshift(abs(f_ind)), interpolation='none')
#plt.imshow(np.fft.fftshift(np.angle(f_ind)), interpolation='none')
upper = np.max(abs(np.real(f_ind)))

n_scale = 2.
base_n_or = 2.
z_cut = 3

first_lp = (1/(n_scale*3))*upper

#center dist
sd_first_lp  = first_lp / z_cut
init_lp = st.norm(scale=sd_first_lp, loc=0).pdf(r).reshape(np.shape(im))

#the scale filter
ends = np.logspace(np.log2(first_lp), np.log2(upper), n_scale + 1, base = 2)
r_means = np.flipud(np.expand_dims(np.diff(ends)/2 + ends[:-1], axis=1))
r_sd = r_means / z_cut

curv_dict = {}
curv_dict['theta_mean'] = []
curv_dict['theta_sd'] = []
curv_dict['r_mean'] = []
curv_dict['r_sd'] = []



n_angles = (base_n_or)*2**n_scale
d_theta = np.pi / n_angles
angles = np.expand_dims(np.linspace(-np.pi/2, -np.pi/2 + np.pi-d_theta, n_angles), axis=1)
sd_theta = d_theta/(z_cut)
for scale, r_mean in enumerate(r_means):
    angle_subset = angles[::2**scale]
    print(scale)
    curv_dict['theta_mean'].append(angle_subset)
    curv_dict['theta_sd'].append(np.ones(np.shape(angle_subset))*(2**(scale/2))*sd_theta)
    curv_dict['r_sd'].append(np.ones(np.shape(angle_subset))*r_sd[scale])
    curv_dict['r_mean'].append(np.ones(np.shape(angle_subset))*r_mean)

for key in curv_dict.keys():
    curv_dict[key] = np.vstack(curv_dict[key])

norm_r = st.norm(scale=curv_dict['r_sd'], loc=curv_dict['r_mean'])
norm_theta = st.norm(scale=curv_dict['theta_sd'], loc=curv_dict['theta_mean'])
filts = (norm_r.pdf(r)*norm_theta.pdf(theta)).reshape((np.size(curv_dict['r_mean']),) + np.shape(im))
<<<<<<< HEAD

norm_r = st.norm(scale=0.09, loc=0.25)
norm_theta = st.norm(scale=0.05, loc=np.pi/2.)

filts = (norm_r.pdf(r)*norm_theta.pdf(theta)).reshape((np.shape(im)))
=======
#
#norm_r = st.norm(scale=0.09, loc=0.25)
#norm_theta = st.norm(scale=0.12, loc=np.pi/2.)
#
#filts = (norm_r.pdf(r)*norm_theta.pdf(theta)).reshape((np.shape(im)))
>>>>>>> 87cedd21897d1da1f5d426956623a511dbed1890


print(np.hstack([curv_dict['r_mean'], curv_dict['r_sd'],
                np.rad2deg(curv_dict['theta_mean']),
                np.rad2deg(curv_dict['theta_sd'])]))


nfilts = np.size(filts,0)
for ind in range(1,np.size(filts,0)):
    plt.subplot(1, nfilts, ind)
    plt.imshow(np.fft.fftshift(abs(filts[ind,:,:])), interpolation ='none')


<<<<<<< HEAD
plt.subplot(2,1,2)
plt.imshow(np.fft.fftshift(abs(filts)), interpolation ='none')


'''
plt.subplot(3,1,3)
plt.plot(np.log10(abs(kern[:,0])))


plt.subplot(1, 2,1)
kern = np.real(np.fft.ifft2(filts[0,:,:]))
plt.imshow(np.fft.fftshift(kern), interpolation ='none')

plt.subplot(1,2, 2)
allfilts = np.sum(filts,axis=0)
plt.imshow(np.fft.fftshift(abs(allfilts)), interpolation ='none')
=======
#
#plt.subplot(2,1,1)
#kern = np.real(np.fft.ifft2(filts[:,:]))
#plt.imshow(np.fft.fftshift(kern), interpolation ='none')
#
#plt.subplot(2,1,2)
#plt.imshow(np.fft.fftshift(abs(filts)), interpolation ='none')
#
#
#plt.subplot(1, 2,1)
#kern = np.real(np.fft.ifft2(filts[0,:,:]))
#plt.imshow(np.fft.fftshift(kern), interpolation ='none')
#
#plt.subplot(1,2, 2)
#allfilts = np.sum(filts,axis=0)
#plt.imshow(np.fft.fftshift(abs(allfilts)), interpolation ='none')
>>>>>>> 87cedd21897d1da1f5d426956623a511dbed1890

#plt.imshow(np.fft.fftshift(np.squeeze(abs(theta_wind[1,0,:,:]))))

#
