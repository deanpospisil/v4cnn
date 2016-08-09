# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:48:04 2016

@author: dean
"""
import sys
import numpy as np
import scipy.io as  l
import scipy
import scipy as sc
import matplotlib.pyplot as plt
import os
import pickle

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'v4cnn/common')
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'

import d_misc as dm
import d_img_process as imp
from scipy import ndimage

def get_center_boundary(x, y):
    minusone = np.arange(-1, np.size(x)-1)
    A = 0.5*np.sum(x[minusone]*y[:] - x[:]*y[minusone])
    normalize= (1/(A*6.))
    cx = normalize * np.sum( (x[minusone] + x[:] ) * (x[minusone]*y[:] - x[:]*y[minusone]) )
    cy = normalize * np.sum( (y[minusone] + y[:] ) * (x[minusone]*y[:] - x[:]*y[minusone]) )
    return cx, cy

def center_boundary(s):
    #centroid, center of mass, https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    for ind in range(len(s)):
        y = s[ind][:, 1]
        x = s[ind][:, 0]
        cx, cy = get_center_boundary(x,y)
        s[ind][:, 0] = x - cx
        s[ind][:, 1] = y - cy

    return s

def scale_center_boundary_for_mat(s, img_n_pix, frac_of_image, max_ext):
    scale = (img_n_pix*frac_of_image)/(max_ext*2.)
    tr = np.round(s*scale)
    tr[:, 0] = tr[:, 0] + img_n_pix/2.
    tr[:, 1] = tr[:, 1] + img_n_pix/2.

    return tr

def boundary_to_mat_by_round(s, img_n_pix, fill=True):
    im = np.zeros((img_n_pix, img_n_pix))
    tr = s.astype(int)

    #conversion of x, y to row, col
    im[(img_n_pix-1)-tr[:, 1], tr[:, 0]] = 1

    if fill:
        im = ndimage.binary_fill_holes(im).astype(int)
#        if not im[tuple(np.median(tr,0))] == 1:
#            raise ValueError('shape not bounded')
    return im


def imgStackTransform(imgDict, shape_boundary):

    n_imgs = np.size(imgDict['shapes'], 0)
    trans_stack = []
    for ind in range(n_imgs):

        transformed_boundary = shape_boundary[imgDict['shapes'][ind]]

#        if 'blur' in imgDict:
#            transformed_boundary = fft_gauss_blur_img( transformed_boundary, imgDict['blur'][ind], std_cut_off = 5 )

        if 'scale' in imgDict:
            if 1 < imgDict['scale'][ind]:
                transformed_boundary = fftDilateImg(transformed_boundary, imgDict['scale'][ind])
            else:
                orig_size = transformed_boundary.shape[0]
                transformed_boundary = zoom(transformed_boundary, imgDict['scale'][ind])
                transformed_boundary = centeredPad(transformed_boundary, orig_size, orig_size)

#        if 'rot' in imgDict:
#            transformed_boundary = scipy.misc.imrotate(transformed_boundary, imgDict['rot'][ind], interp='bilinear')

        if 'x' and 'y' in imgDict:
            x = imgDict['x'][ind]
            y = imgDict['y'][ind]
            transformed_boundary = translateByPixels(transformed_boundary, x, y)

        elif 'x'  in imgDict:
            x = imgDict['x'][ind]
            transformed_boundary = translateByPixels(transformed_boundary, x, np.zeros(np.shape(x)))

        elif 'y'  in imgDict:
            y = imgDict['y'][ind]
            transformed_boundary = translateByPixels(transformed_boundary, x, np.zeros(np.shape(x)))

        transformed_boundary

        trans_stack.append(transformed_boundary)
    trans_stack = np.array(trans_stack)
    return trans_stack
#im_ids = [int(re.findall('\d+[.npy]', fn)[0][:-1]) for fn in stack_desc['img_paths']]
def img_info(base_stack, shape_ids):

    #area
    im_area = np.array([np.sum(img>0)/np.prod(np.shape(img)).astype(float)
                for img in base_stack if np.sum(img>0)>0])
    #upmost row, downmost row, leftmost col, rightmost col
    im_edge = np.array([[np.nonzero(img.sum(1))[0][0], np.nonzero(img.sum(1))[0][-1],
                np.nonzero(img.sum(0))[0][0], np.nonzero(img.sum(0))[0][-1]]
                for img in base_stack if np.sum(img>0)>0])
    im_edge = np.array(im_edge)
    im_power = [np.sum(img**2) for img in base_stack if np.sum(img>0)>0]

    var_names = [ 'area','power', 'up', 'down', 'left', 'right']
    img_vars = np.column_stack([im_area, im_power,
                                im_edge[:,0], im_edge[:,1], im_edge[:,2], im_edge[:,3]])
    im_info = pd.DataFrame(img_vars, columns=var_names, index=shape_ids)
    return im_info
def biggest_x_y_diff(shapes):
    best_max = np.array([0, 0])
    best_min = np.array([0, 0])
    for n_boundary, boundary in enumerate(boundaries):
        max_candidate = boundary.max(axis=0)
        best_max = np.max(np.vstack([best_max, max_candidate]), axis=0)

        min_candidate = boundary.min(axis=0)
        best_min = np.min(np.vstack([best_min, min_candidate]), axis=0)
    return max(best_max-best_min)

#    os.chdir( saveDir + baseImageList[0])
mat = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')
s = np.array(mat['shapes'][0])
boundaries = center_boundary(s)
##adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321]
#a = np.hstack((range(14), range(18,318)))
#a = np.hstack((a, range(322, 370)))
#s = s[a]


img_n_pix = 227
max_pix_width = 24.
#boundaries = boundaries * (max_pix_width/biggest_x_y_diff(boundaries))
#biggest_diff = biggest_x_y_diff(boundaries)
#boundaries = boundaries + img_n_pix/2.

scale = (max_pix_width/biggest_x_y_diff(boundaries))
shape_ids = range(-1, 370)
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(
                                                 shapes=shape_ids,
                                                 blur=None,
                                                 scale=(scale, scale, 1),
                                                 x=(img_n_pix/2., img_n_pix/2., 1),
                                                 y=(img_n_pix/2., img_n_pix/2., 1))
imgDict = stim_trans_cart_dict

trans_stack = imgStackTransform(imgDict, boundaries)


base_stack=[]
fill = True
for n_boundary, boundary in enumerate(boundaries):
    base_stack.append(255. * boundary_to_mat_by_round(boundary, img_n_pix,
                                       fill=fill))



#shape_ids = range(0, 370)
#im_info = img_info(base_stack, shape_ids)
#
#smallest_width = (im_info['right'] - im_info['left']).min()
#widest_width = (im_info['right'] - im_info['left']).max()
#
#largest_row_extent = (im_info['down'].max() - im_info['up'].min())
#
#print(widest_width)
#
#plt.imshow(base_stack[1][:,:].squeeze())