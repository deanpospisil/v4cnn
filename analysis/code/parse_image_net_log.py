# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:26:06 2016

@author: dean
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
import xarray as xr
f = open(top_dir + '/data/image_net/imagenet_log_May_21.txt', 'r')
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
                if not line==[]]):


plt.plot(iteration, (loss-np.mean(loss))/10)
plt.plot(iteration, lr*10)
plt.plot(np.arange(0,len(acc)*1000, 1000), acc)

trans= [(-7, 7, 15), (7, 7, 15), (-50, 48, 50), (-50, 48, 50)]
scales = [1, 0.45, 1, 0.45]
for x, scale in zip(trans,scales):