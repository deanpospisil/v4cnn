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
plt.close('all')
#f = open(top_dir + '/data/image_net/imagenet_log_May_21.txt', 'r')
#f = open(top_dir + '/data/image_net/imagenet_log_training_June21_fromsnapshot.txt', 'r')
#f = open(top_dir + '/data/image_net/imagenet_log_training_June19.txt', 'r')
#f = open(top_dir + '/data/image_net/imagenet_log_training_June27.txt', 'r')
#f = open(top_dir + '/data/image_net/imagenet_log_training_July11th_faithful.txt', 'r')

#all reps
filenames = [
##'imagenet_log_training_july13th_rep_scrib_aws.txt',
##'imagenet_log_training_july13th_rep_scrib_aws2.txt',
##'imagenet_log_training_july13th_rep_scrib.txt',
##'imagenet_log_training_july13th_rep_scrib2.txt',
##'imagenet_log_training_july13th_rep_scrib_aws2_long.txt',
#'imagenet_log_May_21.txt',
#'imagenet_log_training_july14th_rep_scrib_batch100.txt',
##'imagenet_log_training_June14.txt',
#'imagenet_log_training_June19.txt',
##'imagenet_log_training_June21_fromsnapshot.txt',
#'imagenet_log_training_June27.txt',

'log.txt'
]
legstr = ['reduced batch b=60 (GTX770)','reduced batch b=100 (GTX770)',
'june19th b=227 (aws)', 'correct_mean orig_proto b=256 (aws)', 'incorrect_mean orig proto b=256 (aws)', 'donahue reference b=256']

#subtracting mean works
filenames = [
'imagenet_log_training_july19_correct_mean.txt',
'imagenet_log_training_july19_incorrect_mean.txt'
]
legstr = ['calculated mean','caffe hosted mean']

##scribonius also goes down, and aws tracks scribonius really well
#filenames = [
###'imagenet_log_training_july13th_rep_scrib_aws.txt',
###'imagenet_log_training_july13th_rep_scrib_aws2.txt',
###'imagenet_log_training_july13th_rep_scrib.txt',
###'imagenet_log_training_july13th_rep_scrib2.txt',
#'imagenet_log_training_july13th_rep_scrib_aws2_long.txt',
#'imagenet_log_May_21.txt',
#'imagenet_log_training_july14th_rep_scrib_batch100.txt',
###'imagenet_log_training_June14.txt',
#'imagenet_log_training_June19.txt',
###'imagenet_log_training_June21_fromsnapshot.txt',
###'imagenet_log_training_June27.txt',
#
#]
#legstr = ['aws replicate scrib', 'scrib may 21st', 'scrib batch 100 fails', 'june 19 aws 227', 'ref 256']

#reinit experiment
#filenames = [
#'imagenet_log_training_july14th_rep_scrib_batch100.txt',
#'imagenet_log_training_july14th_rep_scrib_batch100_reinit.txt',
#]

def matchlen(a,b):
    if len(a)>len(b):
        a = a[:len(b)]
    elif len(a)<len(b):
        b = b[:len(a)]
    return a, b


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


#f = open(top_dir + '/data/image_net/bvlc_reference_alexnet_train_log.txt', 'r')
#log = f.readlines()
#acc = np.array([np.double(re.split(' ', line[0])[1]) for line in
#                [re.findall('#0: \d+.\d+', line) for line in log]
#                if not line==[]])
#iter_loss = [re.findall('Iteration \d+.+loss = \d+.\d+', line) for line in log
#            if not re.findall('Iteration \d+.+loss = \d+.\d+', line)==[]]
#loss = np.array([np.double(re.split(' ',line[0])[-1]) for line in iter_loss])[:-1]
#iteration = np.array([np.double(re.split(' ', re.split(', ',line[0])[0])[1] ) for line in iter_loss])[:-1]
#
#plt.plot(range(0,len(acc)*1000,1000), acc)
plt.legend(legstr, loc=0)


#plt.subplot(212)
#iter_loss, loss = matchlen(iteration, loss)
#plt.plot(iter_loss, loss, alpha=0.5)

