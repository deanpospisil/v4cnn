# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:07:09 2016

@author: dean
"""


import os, sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)
    


import matplotlib.pyplot as plt
import d_img_process as imp 
import matplotlib.cm as cm
    
img_dir =  cwd  +'/images/baseimgs/PC370/'   
stack, stack_desc = imp.load_npy_img_dirs_into_stack( img_dir )

stack, stack_desc = imp.load_npy_img_dirs_into_stack( img_dir )

plt.imshow(stack[1,:,:])
trans_stack = imp.imgStackTransform( {'shapes':[1,1]}, stack )

plt.close('all')
plt.imshow(trans_stack[1,:,:],cmap = cm.Greys_r, interpolation = 'none')

