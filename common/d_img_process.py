# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:02:26 2015

@author: dean
"""
import scipy.signal as sig
import os
import scipy
import warnings
import numpy as np
pi=np.pi
import scipy as sc
from scipy import interpolate

def cart_to_polar_2d_angles(imsize, sample_rate_mult):

    #get polar resampling coordinates
    n_pix = int(imsize/2.)
    npts_angle = int(np.pi * 2 * n_pix) * sample_rate_mult
    angles_vec = np.linspace(0, 2*np.pi, npts_angle)
    return angles_vec
    
def cart_to_polar_2d_lin(im, sample_rate_mult):
    #take an image, im, and unwrap it into polar coords
    x = np.arange(im.shape[1])
    y = np.arange(im.shape[0])
    f = interpolate.interp2d(x, y, im, kind='linear')

    #get polar resampling coordinates
    n_pix = int(np.size(im,0)/2.)
    npts_mag = np.ceil(np.size(im, 0) / 2.)*sample_rate_mult
    npts_angle = int(np.pi * 2 * n_pix)*sample_rate_mult

    angles_vec = np.linspace(0, 2*np.pi, npts_angle)
    magnitudes_vec = np.linspace(0, n_pix, npts_mag)
    angles, magnitudes = np.meshgrid(angles_vec, magnitudes_vec)
    xnew = (magnitudes * np.cos(angles)+n_pix).ravel()
    ynew = (magnitudes * np.sin(angles)+n_pix).ravel()
    f = interpolate.RegularGridInterpolator((x, y), im, method='linear')

    pts = np.fliplr(np.array([xnew, ynew]).T)
    im_pol = f(pts).reshape(npts_mag, npts_angle)
    im_pol = im_pol * magnitudes*2*np.pi

    return im_pol

def cart_to_polar_2d_lin_broad(im, sample_rate_mult):
    #take a stack of images, image index first, and do it to all of them
    cut = [cart_to_polar_2d_lin(im_cut, sample_rate_mult) for im_cut in  im]
    return cut

def circ_cor(pol1, pol2, sample_rate_mult=2):


    cross_cor = np.fft.ifft(np.fft.fft(pol1, axis=1) *
                        np.fft.fft(np.fliplr(pol2), axis=1),
                        axis=1)
    sum_over_r = np.sum(np.real(cross_cor), axis=0)

    return sum_over_r
def saveToPNGDir(directory, fileName, img):
    import os
    if not os.path.isdir(directory):
        os.mkdir(directory)

    sc.misc.imsave( directory  + fileName + '.png', img)



def getfIndex(nSamps, fs):

    f = np.fft.fftfreq( nSamps, 1./fs)
#    nSamps=np.double(nSamps)
#    fs=np.double(fs)
#    nyq = fs/2
#    df = fs / nSamps
#    f = np.arange(nSamps) * df
#    f[f>nyq] = f[f>nyq] - nyq*2
    return f

def get2dCfIndex(xsamps,ysamps,fs):
    fx, fy = np.meshgrid( getfIndex(xsamps,fs), getfIndex(ysamps,fs) )
    C = fx + 1j * fy
    return C

def fft2Interpolate(coef, points, w):
    basis = np.exp( 1j * 2 * pi * ( points[0,0] * w[0] + points[0,0] * w[1] ))
    nPoints = np.size(points)/2
    intrpvals = np.zeros( nPoints, 'complex')
    for ind in xrange(nPoints):
        basis[:,:] = np.exp( 1j * 2 * pi * ( points[0, ind] * w[0] + points[1, ind] * w[1] ) )
        intrpvals[ind] = np.sum(  coef * basis  )
    return intrpvals


def translateByPixels(img,x,y):
    x = int(np.round(x))
    y = int(np.round(y))
    newImg= np.zeros(np.shape(img))
    nrows= np.size(img,0)
    ncols= np.size(img,1)
    r , c = np.meshgrid( range(nrows), range(ncols) );

    newrow = r-y
    newcol = c+x

    valid = (newrow<nrows) & (newcol<ncols) & (newcol>=0) & (newrow>=0)
    r =  r[valid]
    c =  c[valid]
    newrow = newrow[valid]
    newcol = newcol[valid]

    newImg[newrow,newcol] = img[r,c]

    return newImg

#def FT
def FTcutToNPixels(dR,dC,mat):
    nRows = np.size( mat, 0)
    nCols = np.size( mat, 1 )

    rCutT=np.ceil(dR/2)+1
    rCutB=nRows-np.floor(dR/2)+1;

    cCutL=np.ceil(dC/2)+1#the left column cut off
    cCutR=nCols-np.floor(dC/2)+1;#the right column cut off

    #take the pieces and put them into a smaller image
    top = np.concatenate((mat[:rCutT, :cCutL ], mat[ :rCutT, cCutR: ]),1)
    bottom = np.concatenate((mat[ rCutB:, :cCutL], mat[rCutB:,cCutR:]),1 )
    mat = np.concatenate((top,bottom),0)

    return mat

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


def guassianDownSampleSTD( oldSize, newSize,  stdCutOff, fs ):
    #choose your low pass filter for downsampling based off some STD
    #if you choose 1 std 33% of energy chopped of when downsampling
    #if you choose 3 0.1 %
    oldSize, newSize, stdCutOff, fs = np.double([oldSize, newSize, stdCutOff, fs])

    return ((fs)*(newSize/oldSize))/stdCutOff

#def fftDilateImg

def fft_resample_img(img, nPix, std_cut_off = None):
    #this is only for square images

    if not std_cut_off is None:
        oldSize = np.size(img,0)
        std = guassianDownSampleSTD( oldSize, nPix,  stdCutOff, oldSize )
        sr = sig.resample(img, nPix, window = ('gaussian', std))
        sr = sig.resample(sr, nPix, window = ('gaussian', std ), axis = 1)
    else:
        sr = sig.resample( img, nPix, window = ('boxcar') )
        sr = sig.resample( sr, nPix, window = ('boxcar'), axis = 1)

    return sr

def fft_gauss_blur_img( img, scale, std_cut_off = 5):

    old_img_size = img.shape[0]
    new_img_size = np.round( img.shape[0]*scale )

    std = guassianDownSampleSTD( old_img_size , new_img_size,  std_cut_off, old_img_size )

    sr = sig.resample( img, old_img_size, window = ('gaussian', std) )
    sr = sig.resample( sr, old_img_size, window = ('gaussian', std), axis=1)

    return sr

def fftDilateImg( img, dilR ):
    #just for square images for now
    nPix= np.array(np.shape(img))
    n = np.round(nPix*dilR)
    efRatioX= n[1]/np.double(nPix[1])
    efRatioY= n[1]/np.double(nPix[1])

    if np.double(n[1]/nPix[0]) != n[0]/ np.double(nPix[0]):
        warnings.warn( 'There will be a small distortion, percent '+ str(100*(efRatioY/efRatioX) ))
    if dilR<=0 or dilR>1:
        warnings.warn('No dilations less than or equal to 0, or dilations over 1.')

    temp = fft_resample_img(img, n[0], std_cut_off = None )

    if (np.size(temp)>np.size(img)):

        dilImg = centeredCrop(temp, nPix[0], nPix[1])

    else:
        dilImg = centeredPad(temp, nPix[0], nPix[1])

    return dilImg


def imgStackTransform(imgDict, shape_img):

    n_imgs = np.size( imgDict['shapes'] , 0 )
    trans_stack = []
    for ind in range( n_imgs ):

        trans_img = shape_img[imgDict['shapes'][ind]]

        if 'blur' in imgDict:
            trans_img = fft_gauss_blur_img( trans_img, imgDict['blur'][ind], std_cut_off = 5 )

        if 'scale' in imgDict:
            trans_img = fftDilateImg( trans_img, imgDict['scale'][ind] )

        if 'rot' in imgDict:
            trans_img = scipy.misc.imrotate( trans_img, imgDict['rot'][ind], interp='bilinear')

        if 'x' and 'y' in imgDict:
            x = imgDict['x'][ind]
            y = imgDict['y'][ind]
            trans_img = translateByPixels(trans_img, x, y)

        elif 'x'  in imgDict:
            x = imgDict['x'][ind]
            trans_img = translateByPixels(trans_img, x, np.zeros(np.shape(x)))

        elif 'y'  in imgDict:
            y = imgDict['y'][ind]
            trans_img = translateByPixels(trans_img, x, np.zeros(np.shape(x)))

        trans_stack.append(trans_img)
    trans_stack = np.array(trans_stack)
    return trans_stack


def load_sorted_dir_numbered_fnms_with_particular_extension(the_dir, extension):
    #takes a directory and an extension, and loads up all file names with that extension,
    #attempting to sort them by number, gives back all the sorted file names
    dir_filenames = os.listdir(the_dir)
    file_names = [ file_name for file_name in dir_filenames if file_name.split('.')[-1] == extension ]

    file_names = sorted( file_names, key = lambda num : int(num.split('.')[0]) )

    return file_names

def load_npy_img_dirs_into_stack( img_dir ):
    #given a directory, loads all the npy images in it, into a stack.
    stack_descriptor_dict = {}
    img_names = load_sorted_dir_numbered_fnms_with_particular_extension( img_dir , 'npy')

    #will need to check this for color images.
    stack_descriptor_dict['img_paths'] = [ img_dir + img_name for img_name in img_names ]
    stack = np.array([ np.load( full_img_name ) for full_img_name in stack_descriptor_dict['img_paths'] ], dtype = float)

    #to do, some descriptor of the images for provenance: commit and input params for base shape gen
    #stack_descriptor_dict['base_shape_gen_inputs'] = [ img_dir + img_name for img_name in img_names ]

    return stack, stack_descriptor_dict


##check the dilation function
#import matplotlib.pyplot as plt
#plt.close('all')
#
#img = np.zeros(10,10)
#img = img[0,:,:]
#rImg = fftResampleImg(img, 222, 4 )
#
#rImg = rImg - np.min(rImg)
#srImg = (rImg/np.max(rImg))*255
#
#plt.subplot(311)
#plt.imshow(img, interpolation = 'none', cmap = plt.cm.Greys_r )
#plt.title('Original Image')
#
#plt.subplot(312)
#plt.imshow(srImg, interpolation = 'none', cmap = plt.cm.Greys_r  )
#plt.title('Downsampled Image')
#
#cImg = fftDilateImg(srImg, 0.2 )
#plt.subplot(313)
#plt.imshow(cImg, interpolation = 'none', cmap = plt.cm.Greys_r  )
#plt.title('Shrunk Downsampled Image')
