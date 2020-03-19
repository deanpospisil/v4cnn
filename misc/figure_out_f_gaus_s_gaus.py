#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:32:43 2017

@author: dean
"""
#def gausswin(x, sigma):
#    #alpha
#    alpha = (N-1)/(2*sigma)
#    np.exp(0.5*())
#    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
import numpy as np
import matplotlib.pyplot as plt
def gaussian(n, sig):
    g = np.exp((-1/2.)*(n/sig)**2.)
    return g

sig = 3
N=64
n = np.arange(-(N-1)/2., (N-1)/2. + 1, 1)
y = gaussian(n, sig)
plt.subplot(211)
plt.plot(n, y)


nfft = N
freq = np.arange(-np.pi, np.pi, 2*np.pi/nfft)

wdft = np.fft.fft(y, int(nfft))
plt.subplot(212)
plt.scatter(freq/np.pi, np.fft.fftshift(np.abs(wdft)), s=2)

ydft = gaussian(n, 1./(sig*(2*np.pi/N)))*(sig*(2*np.pi)**0.5)
#ydft = gaussian(freq, 1./sig)*(sig*(2*np.pi)**0.5)
plt.plot(freq/np.pi, ydft)


#%%
s = np.linspace(0.1, 3, 1000) #size of image in meters
plt.figure()
d = 1 #meters from image
img_size = 227 #npixels across
rf_size = 0.5 #size of v1 rf in degrees
img_deg_vangle = np.rad2deg(2*np.arctan(s/2.))
npix = rf_size * ((img_size/np.rad2deg(2*np.arctan(s/2.))))
plt.subplot(121)

plt.plot(s, npix)
plt.xlabel('Size of Image in Meters')
plt.ylabel('Pixels across 0.5 deg (V1 smallest RF)')
plt.subplot(122)
plt.plot(img_deg_vangle/2., npix)
plt.xlabel('Image Eccentricity')

#%%
plt.figure()
plt.scatter(np.linspace(-85./2, 85./2, 85), 85*[0.5,], s=5)

#%%
r = 5
ve = 30
N = 227
a = 7/50
b = 0.6

vi = (ve*r-b*N)/(a*N + r)


R_back = (a*vi + b)*(N/(ve-vi))
