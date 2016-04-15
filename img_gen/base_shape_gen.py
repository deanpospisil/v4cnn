# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:04:21 2015

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

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'net_code/common')
sys.path.append( top_dir + 'xarray/')

import d_curve as dc
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

def scale_center_boundary_for_mat(s, n_pix_per_side, frac_of_image, max_ext):
    scale = (n_pix_per_side*frac_of_image)/(max_ext*2.)
    tr = np.round(s*scale)
    
    cx, cy = get_center_boundary(tr[:, 0], tr[:, 1])
    tr[:, 0] = tr[:, 0] + n_pix_per_side/2.- cx + 1
    tr[:, 1] = tr[:, 1] + n_pix_per_side/2.- cy + 1
    
    return tr

def boundary_to_mat_by_round(s, n_pix_per_side, frac_of_image, max_ext, fill=True):
    im = np.zeros((n_pix_per_side, n_pix_per_side))
    tr = scale_center_boundary_for_mat(s, n_pix_per_side, frac_of_image, max_ext)
    tr = s.astype(int)
        
    #conversion of x, y to row, col
    im[(n_pix_per_side-1)-tr[:, 1], tr[:, 0]] = 1

    if fill:
        im = ndimage.binary_fill_holes(im).astype(int)

#        if not im[tuple(np.median(tr,0))] == 1:
#            raise ValueError('shape not bounded')
    return im


def boundary_to_mat_via_plot(boundary, n_pix_per_side=227, frac_of_img=1, fill=True):
    n_pix_per_side_old = n_pix_per_side
    if frac_of_img > 1:
        n_pix_per_side = round(n_pix_per_side * frac_of_img)
    plt.close('all')
    inchOverPix = 2.84/227. #this happens to work because of the dpi of my current screen. 1920X1080
    inches = inchOverPix*n_pix_per_side
    if inches<0.81:
        print( 'inches < 0.81, needed to resize')
        tooSmall = True
        inches = 0.85

    fig = plt.figure(figsize = ( inches, inches ))#min size seems to be 0.81 in the horizontal, annoying
    plt.axis( 'off' )
    plt.gca().set_xlim([-1, 1])
    plt.gca().set_ylim([-1, 1])
    if fill is True:
        line = plt.Polygon(boundary, closed=True, fill='k', edgecolor='none', fc='k')
    else:
        line = plt.Polygon(boundary, closed=True, fill='k', edgecolor='k',fc='w')
    plt.gca().add_patch(line)
    fig.canvas.draw()
    data1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data2 = data1.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    data2[data2 == data2[0,0,0]] = 255
    ima = - (data2 - 255)[:,:,0]
    ima = imp.centeredCrop(ima, n_pix_per_side_old, n_pix_per_side_old)
    if (not np.size( ima, 0 ) == n_pix_per_side_old) or (not np.size(ima,1 ) == n_pix_per_side_old):
        print('had to resize')
        ima = scipy.misc.imresize(ima, (n_pix_per_side_old, n_pix_per_side_old), interp='cubic', mode=None)
    return ima


def save_boundaries_as_image(imlist, save_dir, cwd, max_ext, n_pix_per_side=227,
                             fill=True, require_provenance=False,
                             frac_of_image=1, use_round=True):
    dir_filenames = os.listdir(save_dir)
    #remove existing files
    for name in dir_filenames:
        if 'npy' in name or 'png' in name or 'pickle' in name:
            os.remove(save_dir + name)
    if require_provenance is True:
        #commit the state of the directory and get is sha identification
        sha = dm.provenance_commit(top_dir)
        #now save that identification with the images
        sha_file = save_dir + 'sha1'
        with open( sha_file + '.pickle', 'wb') as f:
            pickle.dump( sha, f )
    imlist = scaleBoundary (imlist, frac_of_image)
    for n_boundary, boundary in enumerate(imlist):
        print(n_boundary)
        if not use_round:
            im = boundary_to_mat_via_plot(boundary, n_pix_per_side,
                                          frac_of_image, fill=fill)
        else:
            im = boundary_to_mat_by_round(boundary, n_pix_per_side,
                                          frac_of_image, max_ext, fill=fill)

        sc.misc.imsave(save_dir + str(n_boundary) + '.bmp', im)
        np.save(save_dir + str(n_boundary), im)

def scaleBoundary(s, fracOfImage):
    if fracOfImage>1:
        fracOfImage =1
    #get the furthest point from the center
    ind= -1
    curmax = 0
    for sh in s:
        ind+=1
        testmax = np.max(np.sqrt( np.sum(sh**2, 1) ))
        if curmax<testmax:
            curmax = testmax
    #scale the shape so the furthest point from the center is some fraction of 1
    scaling=curmax/fracOfImage
    for ind in range(len(s)):
        s[ind] = s[ind]/scaling

    return s

def pixel_arc(pix_ref, pix_n, radius, arclen, npoints):
    cmplx = np.array([1, 1j])
    pix_dir = pix_ref - pix_n
    ang = np.angle(np.sum(pix_dir*cmplx))

    shifts = np.exp(np.linspace(-arclen/2, arclen/2, npoints)*1j)
    center = np.exp(ang*1j) #rotate by 180 degrees
    cpoints = radius*(shifts*center)
    rpoints = np.round(np.array([np.real(cpoints), np.imag(cpoints)]).T)
    return rpoints

def arc_neigbor_max_2d(im, cur_ind, around):
    cur_ind = np.array(cur_ind)
    aind = [cur_ind + shift for shift in around
            if ((cur_ind + shift)>=0).any() and
            ((cur_ind[0] + shift[0])<im.shape[0]) and
            ((cur_ind[1] + shift[1])<im.shape[1])]
    min_ind = aind[np.argmax([im[i[0], i[1]] for i in aind])]
    return min_ind

def trace_edge(im, scale, radius, npts = 100, maxlen = 1000):
    effective_radius = radius/scale
    arclen = 2*np.arcsin(effective_radius)
    if arclen>np.pi or np.isnan(arclen):
        arclen = np.pi
    #resample image
    ims = imp.fft_resample_img(im, im.shape[1]*scale, std_cut_off = None)
    temp = np.gradient(ims)#get the gradient of the image
    d = temp[0]**2 + temp[1]**2

    #start at first peak
    first_peak = np.array(np.unravel_index(np.argmax(d), d.shape))
    around = pixel_arc(first_peak, first_peak, radius, np.pi*2, npts)
    cur_peak = arc_neigbor_max_2d(d, first_peak, around)

    line = []
    line.append(first_peak)
    line.append(cur_peak)
    i = 0
    #append to line, till too long, or wraps around
    while not (line[0]==cur_peak).all() and len(line)<maxlen:
        around = pixel_arc(line[i+1], line[i], radius, arclen, npts)
        cur_peak = arc_neigbor_max_2d(d, line[i+1], around)
        i+=1
        line.append(cur_peak)
        if np.sum(np.array(line[0]-cur_peak)**2)**0.5<(radius-1):
            break

    return np.array(line), d


#first get an imageset
#img_dir = top_dir + 'images/baseimgs/PC370/'
#stack, stack_descriptor_dict = imp.load_npy_img_dirs_into_stack(img_dir)
#im = stack[0,:,:]

#generate base images
'''
saveDir = top_dir + 'net_code/images/baseimgs/'
dm.ifNoDirMakeDir(saveDir)

baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
baseImage = baseImageList[0]

frac_of_image = 1
dm.ifNoDirMakeDir(saveDir + baseImage +'/')

if baseImage is baseImageList[0]:

#    os.chdir( saveDir + baseImageList[0])
    mat = l.loadmat(top_dir + 'net_code/img_gen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])

elif baseImage is baseImageList[1]:
    nPts = 1000
    s = dc.make_n_natural_formlets(n=1000,
                nPts=nPts, radius=1, nFormlets=32, meanFormDir=np.pi,
                stdFormDir=2*np.pi, meanFormDist=1, stdFormDist=0.1,
                startSigma=3, endSigma=0.1, randseed=1, min_n_pix=64,
                frac_image=frac_of_image)
elif baseImage is baseImageList[2]:
    #    os.chdir( saveDir + baseImageList[0])
    mat = l.loadmat(top_dir + 'net_code' + '/img_gen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])
    #adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321]
    a = np.hstack((range(14), range(18,318)))
    a = np.hstack((a, range(322, 370)))
    s = s[a]

elif baseImage is baseImageList[3]:
    print('to do')


s = center_boundary(s)
max_ext = np.max([np.max(np.abs(a_s)) for a_s in s])

save_boundaries_as_image(s, saveDir + baseImage + '/', top_dir, max_ext,
                         n_pix_per_side=227, fill=True, require_provenance=False,
                         frac_of_image=frac_of_image, use_round=True)



ashape = s[0]


x = s[0][:, 0]
y = s[0][:, 1]
plt.close('all')
plt.plot(x, y)
max_ext = np.max(np.abs(s[0]))
n_pix_per_side = 16.
frac_of_image = 0.5
dists = np.sum((np.diff(s[0].T))**2, axis=0)**0.5

#s = scale_center_boundary_for_mat(s.T, n_pix_per_side, frac_of_image, max_ext)
scale = (n_pix_per_side*frac_of_image)/(max_ext*2.)
dx = np.median(dists)*scale
freqs = np.fft.fftfreq(len(x), dx)
low_pass = np.zeros(np.shape(s[0]))*1j
low_pass[np.abs(freqs)<((0.5**0.5)/2.), :] = 1.

ft = np.fft.fft(s[0], axis=0) * low_pass

lp = np.real_if_close(np.fft.ifft(ft, axis=0))
plt.plot(lp[:, 0], lp[:, 1])

lp = lp * scale
cx, cy = get_center_boundary(lp[:,1], lp[:,0])
lp[:, 1] = lp[:, 1] + (n_pix_per_side/2.0 - cx)
lp[:, 0] = lp[:, 0] + (n_pix_per_side/2.0 - cy)

u_scale = 500
im = boundary_to_mat_by_round(s[0], u_scale, frac_of_image, max_ext)
ims = imp.fft_resample_img(im, n_pix_per_side)
ims = imp.fft_resample_img(ims, u_scale)
import matplotlib.cm as cm
plt.imshow(ims, cmap = cm.Greys_r, interpolation = 'none')
plt.plot(lp[:, 0] * (u_scale/n_pix_per_side), lp[:, 1] * (u_scale/n_pix_per_side),  '#FA8072', linewidth=1.0)
'''

'''
dists = np.sum((np.diff(ashape).T)**2, axis=0)**0.5
xp = np.cumsum(dists)
mindx = np.median(dists)
print(mindx)
x = np.interp(np.linspace(0, xp[-1], nPts), xp, ashape[:, 1])
y = np.interp(np.linspace(0, xp[-1], nPts), xp, ashape[:, 0])
dists = np.sum((np.diff(np.array([x,y])))**2, axis=0)**0.5



print(len(x))

s = center_boundary(s)
s = scaleBoundary (s, fracOfImage)

max_ext = np.max(np.abs(s))
n_pix_per_side = 256.
frac_of_image = 0.5
ashape = scale_center_boundary_for_mat(ashape, n_pix_per_side, frac_of_image, max_ext)

'''

#im = boundary_to_mat_by_round(ashape, n_pix_per_side, frac_of_image, max_ext)


'''
npts = 100
maxlen = 2000
line, d = trace_edge(im, scale, radius, arclen, npts, maxlen)
u_line.append(np.array(line))
u_img.append(d)

scale = 2*256/64.
radius = 5
effective_radius = radius/scale
arclen = 2*np.arcsin(effective_radius)
n_pix_per_side = 64.
frac_of_image = 0.5
ims = imp.fft_resample_img(im, n_pix_per_side, std_cut_off = None)
line, d = trace_edge(ims, scale, radius, arclen, npts, maxlen)
d_line.append(np.array(line))
d_img.append(d)

ims = imp.fft_resample_img(im, im.shape[1]*scale, std_cut_off = None)
temp = np.gradient(ims)#get the gradient of the image
d = temp[0]**2 + temp[1]**2

#start at first peak
first_peak = np.array(np.unravel_index(np.argmax(d), d.shape))
around = pixel_arc(first_peak, first_peak, radius, np.pi*2, npts)
cur_peak = arc_neigbor_max_2d(d, first_peak, around)


line = []
line.append(first_peak)
line.append(cur_peak)
i = 0
#append to line, till too long, or wraps around
while not (line[0]==cur_peak).all() and len(line)<maxlen:
    around = pixel_arc(line[i+1], line[i], radius, arclen, npts)
    cur_peak = arc_neigbor_max_2d(d, line[i+1], around)
    i+=1
    line.append(cur_peak)
    if np.sum(np.array(line[0]-cur_peak)**2)**0.5<(radius+1):
        break


import matplotlib.cm as cm
plt.close('all')
ind = 0
plt.subplot(221)
plt.imshow(ims, cmap = cm.Greys_r, interpolation = 'none')
plt.scatter(d_line[ind][0,1]/scale, d_line[ind][0,0]/scale)
plt.plot(d_line[ind][:,1]/scale, d_line[ind][:,0]/scale)
plt.subplot(222)
plt.imshow(u_img[ind], cmap = cm.Greys_r, interpolation = 'none')
plt.scatter(u_line[ind][0,1], u_line[ind][0,0])
plt.plot(u_line[ind][:,1], u_line[ind][:,0])
plt.subplot(224)
plt.plot(u_line[ind][:,:])
plt.subplot(223)
plt.plot(d_line[ind][:,:])
'''
