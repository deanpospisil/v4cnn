#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:07:36 2017

@author: dean
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
import xarray as xr
plt.close('all')

filenames = ['log_lmdb.txt',]
top_dir = '/home/dean/caffe/models/msnet'

def matchlen(a,b):
    if len(a)>len(b):
        a = a[:len(b)]
    elif len(a)<len(b):
        b = b[:len(a)]
    return a, b


for filename in filenames:
    f = open(top_dir +'/'+ filename, 'r')

    log = f.readlines()


    iter_loss = [re.findall('Iteration \d+.+loss = \d+.\d+', line) for line in log
                if not re.findall('Iteration \d+.+loss = \d+.\d+', line)==[]]
    loss = np.array([np.double(re.split(' ',line[0])[-1]) for line in iter_loss])[:-1]
    iteration = np.array([np.double(re.split(' ', re.split(', ',line[0])[0])[1] ) for line in iter_loss])[:-1]

    lr = [re.findall(' lr = \d+.\d+', line) for line in log]

    lr = np.array([float(re.split(' = ', line[0])[1]) for line in
                    [re.findall(' lr = \d+.\d+| lr = \de-\d+', line) for line in log]
                    if not line==[]])

    acc = np.array([np.double(re.split(' = ', line[0])[1]) for line in
                    [re.findall('accuracy = \d+.\d+', line) for line in log]
                    if not line==[]])
    acc_iter = np.array([int(re.split(',', line[0])[0]) for line in
                    [re.findall('\d+, Testing net', line) for line in log]
                    if not line==[]])

    iter_loss, loss = matchlen(iter_loss, loss)
    acc_iter, acc = matchlen(acc_iter, acc)
    plt.subplot(211)
    plt.plot(acc_iter, acc)
    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.ylim(0,1)

    plt.subplot(212)
    plt.plot(iteration, loss)
    plt.ylabel('loss')

plt.subplot(211)