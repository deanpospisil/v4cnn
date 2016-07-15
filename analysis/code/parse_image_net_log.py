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
top_dir = top_dir + 'v4cnn'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
import xarray as xr
#f = open(top_dir + '/data/image_net/imagenet_log_May_21.txt', 'r')
#f = open(top_dir + '/data/image_net/imagenet_log_training_June21_fromsnapshot.txt', 'r')
#f = open(top_dir + '/data/image_net/imagenet_log_training_June19.txt', 'r')
#f = open(top_dir + '/data/image_net/imagenet_log_training_June27.txt', 'r')
#f = open(top_dir + '/data/image_net/imagenet_log_training_July11th_faithful.txt', 'r')
filenames = ['imagenet_log_training_july13th_rep_scrib_aws.txt',
#'imagenet_log_training_july13th_rep_scrib_aws2.txt',
#'imagenet_log_training_july13th_rep_scrib.txt',
#'imagenet_log_training_july13th_rep_scrib2.txt',
'imagenet_log_training_july13th_rep_scrib_aws2_long.txt',
'imagenet_log_training_july14th_rep_scrib_batch100.txt',
'imagenet_log_training_June14.txt',
'imagenet_log_training_June19.txt',
#'imagenet_log_training_June21_fromsnapshot.txt',
'imagenet_log_training_June27.txt',
'imagenet_log_May_21.txt']

for filename in filenames:
    f = open(top_dir + '/data/image_net/' + filename, 'r')

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
    if len(acc_iter)>len(acc):
        acc_iter = acc_iter[:len(acc)]
    elif len(acc_iter)<len(acc):
        acc = acc[:len(acc_iter)]

    plt.plot(acc_iter, acc)
    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.ylim(0,1)


f = open(top_dir + '/data/image_net/bvlc_reference_alexnet_train_log.txt', 'r')
log = f.readlines()
acc = np.array([np.double(re.split(' ', line[0])[1]) for line in
                [re.findall('#0: \d+.\d+', line) for line in log]
                if not line==[]])
plt.plot(range(0,len(acc)*1000,1000), acc)

#plt.legend(['aws1','aws2', 'scrib1', 'scrib2', 'aws60 long','scrib100 long','june 14 scrib82','june19th aws 227', 'june21 snap', 'may21_scrib', 'ref'])
plt.legend(['aws1','aws60 long','scrib100 long','june 14 scrib82',
'june19th aws 227', 'June 27 aws 256', 'may21_scrib', 'ref'], loc=0)