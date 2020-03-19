# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:26:15 2018

@author: deanpospisil
"""

import numpy as np
import matplotlib.pyplot as plt
#plt.scatter(dr[:,0], dr[:,1])
#plt.xlim(0,20);plt.ylim(0,20);
#plt.plot([0,20],[0,20])
n_samps = 2
n_neuron = 10
dat = np.zeros(n_samps)
for n in range(n_samps):
    plt.figure(figsize=(4,4))

    dr = np.abs(np.random.normal(loc=0, scale=5, size=(n_neuron,2)))
    #plt.hist(dr.ravel())
    plt.scatter(dr[:,0], dr[:,1])

    tex = np.random.normal(np.abs(dr[:,1]), size=(30,n_neuron))
    shape = np.random.normal(np.abs(dr[:,0]), size=(30,n_neuron))
    plt.figure(figsize=(8,4))

    plt.subplot(121)
    plt.title(np.round(np.corrcoef(np.var(tex,0),np.var(shape,0))[0,1],2))
    plt.scatter(np.var(tex,0),np.var(shape,0))
    for a, x, y in  zip(range(n_neuron),np.var(tex,0),np.var(shape,0)):
        plt.text(x=x, y=y, s=a)
    
    mtex = np.max(tex, 0)
    mshape = np.max(shape, 0)
    mboth = np.max(np.array([mtex,mshape]),0)
    plt.axis('equal')
    plt.xlim(0,2);plt.ylim(0,2)

    n_tex = tex/mboth
    n_shape = shape/mboth
    
    plt.subplot(122)
    vn_tex = np.var(n_tex,0)
    vn_shape = np.var(n_shape,0)
    
    dat[n] = np.corrcoef(vn_tex, vn_shape)[0,1]
    plt.title(np.round(dat[n], 2))
    plt.scatter(vn_tex, vn_shape)
    for a, x, y in  zip(range(n_neuron),vn_tex, vn_shape):
        plt.text(x=x, y=y, s=a)
    plt.axis('equal')

    

#plt.plot(dat)



