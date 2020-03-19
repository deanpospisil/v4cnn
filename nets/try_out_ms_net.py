#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:21:25 2017

@author: dean
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:06:13 2017

@author: dean
"""

from pylab import *

caffe_root = '/home/dean/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import numpy as np
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
os.chdir(caffe_root)
os.chdir('examples')

from caffe import layers as L, params as P
import matplotlib.pyplot as plt


caffe.set_device(0)
caffe.set_mode_gpu()
#%%
### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
#solver = caffe.SGDSolver('/home/dean/caffe/examples/cifar10/cifar10_full_solver.prototxt')
solver = caffe.SGDSolver('/home/dean/caffe/models/msnet/solver.prototxt')

solver.net.params['region1_resmod0_conv'][0].data

#%%
for k, v in solver.net.blobs.items():
    print((k, v.data.shape)) 
batch = 500
#solver.net.forward()  # train net
#solver.test_nets[0].forward()  # test net (there can be more than one)
#plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
#print 'train labels:', solver.net.blobs['label'].data[:8]
#solver.step(1)
#
#plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
#       .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off')
niter = 50000
test_interval = 100
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
output = np.zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    if it % test_interval == 0:
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['accuracy'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / (batch*100.)
        print(correct / (batch*100.))

#%%
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
#%%
def colorize(im):
    #expects rxcxRGB
    im -= np.min(im)
    im /= np.max(im)
    return im
image = 1
a_filter  = 0
n_plot = 29

im = solver.net.blobs['data'].data[image, ...]
plt.figure(figsize=(1,1))
plt.imshow(colorize(np.copy(im).T.swapaxes(0,1)))
for n_conv in range(1,4):
    t = solver.test_nets[0].blobs['conv{}'.format(n_conv + 1)].data
    #conv = np.sum(np.real(np.fft.ifft2(np.fft.fft2(t)**2)[:, :, ...]), (0,1))
    plt.figure(figsize=(40,2))
    for i in range(1, n_plot):
        plt.subplot(1, n_plot, i);
        plt.imshow(t[image, a_filter+i,...]);
        plt.xticks([]);plt.yticks([])
    #plt.tight_layout()
#%%
for im in solver.net.params['conv1'][0].data:
    t = im.T
    t = t.swapaxes(0,1)
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.mean(t,-1));plt.colorbar();
    plt.subplot(212)
    plt.imshow(colorize(np.copy(t)))
    
#%%    
def norm_cov(x, subtract_mean=True):
    
    #if nxm the get cov mxm
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 0, keepdims=True)
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    vnrm = np.linalg.norm(x, axis=0, keepdims=True)
    denominator = np.sum(np.multiply(vnrm.T, vnrm)[diag_inds]) 
    norm_cov = numerator/denominator
    
    return norm_cov

for n_conv in range(0, 5):
    t = solver.net.params['conv{}'.format(n_conv + 1)][0].data
    wt_cov = []
    for a_filt in t:
        a_filt_unrw = a_filt.reshape(a_filt.shape[0], np.product(a_filt.shape[1:]))
        wt_cov.append(norm_cov(a_filt_unrw, subtract_mean=False))
    
    plt.figure()
    plt.hist(wt_cov);plt.xlim(-1,1);
    #conv = np.sum(np.real(np.fft.ifft2(np.fft.fft2(t)**2)[:, :, ...]), (0,1))



