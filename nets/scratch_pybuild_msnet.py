#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:15:18 2017

@author: dean
"""

#%%
import os
import sys
import re
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0] 
sys.path.append(top_dir+ 'v4cnn')
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')
sys.path.append( top_dir + 'nets/')

#import caffe_net_response as cf
import d_misc as dm
import xarray as xr
import apc_model_fit as ac
import d_curve as dc
import d_img_process as imp
import scipy.io as l



def rf_width(kernel_widths, strides):
    rf_width = [1, ]
    strides = np.array(strides)
    kernel_widths = np.array(kernel_widths)

    kernel_widths = np.insert(kernel_widths, 0, 1)
    strides = np.insert(strides, 0, 1)

    stride_prod = np.cumprod(strides)

    for i in range(len(kernel_widths))[1:]:
        rf_width.append(rf_width[i-1] + (kernel_widths[i] - 1) *
                        stride_prod[i-1])

    return rf_width[1:]

def output_sizes(kernel_widths, strides, input_size):
    if not (type(input_size) is type(list())):
        input_size = [input_size,] 
        
    for i in range(len(strides)):
        input_size.append(int(np.ceil((input_size[i] - kernel_widths[i]) 
                         / strides[i] + 1)))
    return input_size[1:]
def layer_txt(net_params):
    # hand a list of net params
    # the first entry is the type of param and the second the value
    # if the value is a list these are subparams,
    # with the first entry the name of those subparams
    nettxt = []
    nettxt.append('layer {')
    for key, val in net_params:
        if type(val) == list:
            nettxt.append('    ' + key + ' {')
            for sub_key, sub_val in val:
                if type(sub_val) == list:
                    nettxt.append('        ' + sub_key + ' {') 
                    for sub_sub_key, sub_sub_val in sub_val:
                        nettxt.append('      ' + sub_sub_key + ': ' + sub_sub_val)
                    nettxt.append('        }')
                else:
                    nettxt.append('      ' + sub_key + ': ' + sub_val)
            nettxt.append('    }')
        
        else:
            nettxt.append('    ' + key + ': ' + val)
    nettxt.append('}')
    return nettxt

def bsr(bottom, bn=True, relu=True):
    layers = []
    if bn:
        sub_name = bottom + '_bn'
        params = [['name', '"'+ sub_name +'"'],
                  ['type', '"BatchNorm"'], 
                  ['bottom', '"'+ bottom +'"'],
                  ['top', '"'+ bottom +'"'],
                  ['batch_norm_param', [['use_global_stats', 'true'], 
                                        ['eps', str(2e-5)],
                 ]]] 
        layers.append(layer_txt(params))
        layer_names.append(sub_name)
        
        sub_name = bottom + '_sc'
        params = [['name', '"'+ sub_name +'"'],
                  ['type', '"Scale"'], 
                  ['bottom', '"'+ bottom +'"'],
                  ['top', '"'+ bottom +'"'],
                  ['scale_param', [['bias_term', 'true']
                 ]]] 
        layers.append(layer_txt(params))
        layer_names.append(sub_name)
        
    if relu:
        sub_name = bottom + '_relu'
        params = [['name', '"' + sub_name + '"'],
                  ['type', '"ReLU"'], 
                  ['bottom', '"' + bottom + '"'],
                  ['top', '"'+ bottom +'"']
                 ] 
        layers.append(layer_txt(params))
        layer_names.append(sub_name)
    return layers, layer_names

def res_mods(name_prefix, bottom, groups, channels, depth):
    #take the prefix of a layer regionX_resmod
    layers = []
    layer_names = []
    top_name = bottom #
    orig_bottom = bottom
    #each loop is one residual module
    for resmod_n in range(1, depth+1):
        
        bottom = top_name #the bottom of the next resmod will be the top of the last
        top_name = name_prefix + str(resmod_n) + '_lin1'
        params = [['name', '"' + top_name + '"'],['type', '"Convolution"'], 
                  ['bottom', '"'+ bottom +'"'],
                  ['top', '"'+ top_name +'"'],
                  ['convolution_param', [['num_output', str(channels)], 
                                         ['kernel_size', '1'],
                                         ['stride', '1'],
                                         ['bias_term', 'false'],
                                         ['group', str(groups)]
                                         ]]] 
        layers.append(layer_txt(params))
        layer_names.append(top_name)
        a_layers, a_layer_names = bsr(top_name)
        layer_names = layer_names + a_layer_names
        layers = layers + a_layers
        
        bottom = top_name
        top_name = name_prefix + str(resmod_n) + '_lin2'
        params = [['name', '"' + top_name + '"'],['type', '"Convolution"'], 
                  ['bottom', '"'+ bottom +'"'],
                  ['top', '"'+ top_name +'"'],
                  ['convolution_param', [['num_output', str(channels)], 
                                         ['kernel_size', '1'],
                                         ['stride', '1'],
                                         ['bias_term', 'false'],
                                         ['group', '1']
                                         ]]] 
        layers.append(layer_txt(params))
        layer_names.append(top_name)
        a_layers, a_layer_names = bsr(top_name, relu=False)
        layer_names = layer_names + a_layer_names
        layers = layers + a_layers
    
        sub_name = name_prefix + str(resmod_n) + '_plus'
        params = [['name', '"' + sub_name + '"'],
              ['type', '"Eltwise"'], 
              ['bottom', '"' + orig_bottom + '"'],
              ['bottom', '"' + top_name + '"'],
              ['top', '"'+ sub_name +'"']
             ] 
        layers.append(layer_txt(params))
        layer_names.append(sub_name)
        
        orig_bottom = sub_name#need to keep track of this original bottom as
        #top name will be used again.
        top_name = sub_name #this eltwise will be the new top, which will be the 
        #the bottom of the next resmod
        
        sub_name = bottom + '_relu'
        params = [['name', '"' + sub_name + '"'],
                  ['type', '"ReLU"'], 
                  ['bottom', '"' + top_name + '"'],
                  ['top', '"'+ top_name +'"']]
        layers.append(layer_txt(params))
        layer_names.append(sub_name)
        

        
    return layers, layer_names
        
#%%
kernel_widths = [2, 4, 4, 3, 2]
strides = [1, 2, 2, 2, 1]
group = 4
d = 16

rw = rf_width(kernel_widths, strides)
fm = output_sizes(kernel_widths, strides, 32)

print(rw)
print(fm)

#%%
bottom = 'data'
input_dim = (1, 3, 32, 32)
kernel_width = 2
stride = 1
channels = 16
groups = 2
depth = 3
n_categories = 10

layers = []
layer_names = []

#creating the data layer
params = [['name', '"data"'],['type', '"Input"'], 
          ['top', '"data"'],
          ['input_param', [['shape', '{dim: ' + str(input_dim[0]) 
                                      + ' dim: ' + str(input_dim[1])
                                      + ' dim: ' + str(input_dim[2])
                                      + ' dim: ' + str(input_dim[3]) + '}'],
                            ]]] 
layers.append(layer_txt(params))
top_name = 'data'
layer_names.append('data')

#conv_resmod(bottom, region, kernel_width, stride, channels, groups, depth):

for kernel_width, stride, region in zip(kernel_widths, strides, range(len(strides))):
    name_prefix = 'region' + str(region+1) + '_resmod' 
    bottom = top_name
    top_name = name_prefix + '0_conv'
    params = [['name', '"' + top_name + '"'],['type', '"Convolution"'], 
              ['bottom', '"'+ bottom +'"'],
              ['top', '"'+ top_name +'"'],
              ['convolution_param', [['num_output', str(channels)], 
                                     ['kernel_size', str(kernel_width)],
                                     ['stride', str(stride)],
                                     ['bias_term', 'false']
                                     ]]] 
    layers.append(layer_txt(params))
    layer_names.append(top_name)
    
    #bnorm, scale, relu,
    a_layers, a_layer_names = bsr(top_name)
    layer_names = layer_names + a_layer_names
    layers = layers + a_layers
    
    #residual modules.
    a_layers, a_layer_names = res_mods(name_prefix, top_name, groups, channels, depth)
    layer_names = layer_names + a_layer_names
    layers = layers + a_layers
    top_name = layer_names[-2]    
#%%
n_categories = 10
#creating the category layer
params = [['name', '"fc"'],['type', '"InnerProduct"'], 
          ['top', '"fc"'],
          ['bottom', '"'+ top_name +'"'],
          ['inner_product_param', [['num_output', str(n_categories)], 
                                   ['weight_filler', [['type', '"xavier"'],]],
                                   ['bias_filler', [['type', '"constant"'], ['value', '0'],]]
                                   ],
            ]]  

layers.append(layer_txt(params))
layer_names = layer_names + ['fc', ]

#creating the category layer
params = [['name', '"prob"'],['type', '"Softmax"'], 
          ['bottom', '"fc"'], ['top', '"prob"']]
layers.append(layer_txt(params))
layer_names = layer_names + ['prob', ]



'''    
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "pool1"
  top: "fc1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}

layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc1"
  top: "prob"
}  
'''
#%%
txt = ''
for layer in layers:
    for line in layer:
        txt = txt + line +'\n'
print(txt)

f = open('ms_net.txt','w')
f.write(txt)
f.close()


