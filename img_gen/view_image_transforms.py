# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:07:09 2016

@author: dean
"""

from scipy.ndimage import zoom
import os, sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)

def centeredPad( img, new_height, new_width):
   width =  np.size(img,1)
   height =  np.size(img,0)

   #if an odd number defaults to putting assym pixel

   hDif = (new_width - width)/2.
   vDif = (new_height - height)/2.

   left = int(np.ceil(hDif))
   top = int(np.ceil(vDif))
   right = int(np.floor(hDif))
   bottom = int(np.floor(vDif))

   pImg = np.pad(img, ( (left, right), (top, bottom) ) ,'constant')
   return pImg



def centeredCrop(img, new_height, new_width):
   width =  np.size(img,1)
   height =  np.size(img,0)

   #if an odd number defaults to putting extra pixel to left and top
   left = np.ceil((width - new_width)/2.)
   top = np.ceil((height - new_height)/2.)
   right = np.floor((width + new_width)/2.)
   bottom = np.floor((height + new_height)/2.)

   cImg = img[top:bottom, left:right]
   return cImg


import matplotlib.pyplot as plt
import d_img_process as imp
import matplotlib.cm as cm

img_dir =  cwd  +'/images/baseimgs/PC370/'
stack, stack_desc = imp.load_npy_img_dirs_into_stack( img_dir )

stack, stack_desc = imp.load_npy_img_dirs_into_stack( img_dir )

plt.imshow(stack[1,:,:])
trans_stack = imp.imgStackTransform( {'shapes':[102,],'scale':[5,]}, stack )

plt.close('all')
plt.imshow(trans_stack[0,:,:],cmap = cm.Greys_r, interpolation = 'nearest')

a = trans_stack[0,:,:]

s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)

#
#b = zoom(a, 0.5)
#
#az=centeredPad(b, a.shape[0], a.shape[0])

#plt.subplot(311)
#plt.imshow(a,cmap = cm.Greys_r, interpolation = 'nearest')
#plt.subplot(312)
#plt.imshow(b,cmap = cm.Greys_r, interpolation = 'nearest')
#
#plt.subplot(313)
#plt.imshow(az,cmap = cm.Greys_r, interpolation = 'nearest')
