# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:02:24 2015

@author: dean
"""
from scipy import interpolate
import warnings
import numpy as np
pi=np.pi

def get_center_boundary(x, y):
    minusone = np.arange(-1, np.size(x)-1)
    A = 0.5*np.sum(x[minusone]*y[:] - x[:]*y[minusone])
    normalize= (1/(A*6.))
    cx = normalize * np.sum((x[minusone] + x[:]) * (x[minusone]*y[:] - x[:]*y[minusone]))
    cy = normalize * np.sum((y[minusone] + y[:]) * (x[minusone]*y[:] - x[:]*y[minusone]))
    return cx, cy

def curveDists(cShape):
    nPts=np.size(cShape)
    dists=np.ones([nPts])*100j
    for ind in range(nPts):
        dists[ind] = abs(cShape[ind] - cShape[ind-1 ])
    return dists

def curve_curvature(cShape):

    nPts=np.size(cShape)
    curvature = np.ones([nPts])*np.nan
    for ind in range(nPts):
        oldDir = (cShape[(ind-1)]) - (cShape[(ind-2)])
        newDir = (cShape[ind]) - (cShape[(ind-1)])
        curvature[ind] = np.angle(newDir * np.conj(oldDir))/(abs(newDir))
    if np.isnan(curvature).any():
        warnings.warn('Did not fill all orientation values')
    return curvature

def curveOrientations(cShape):
    #gets a complex unit vector leading up to each point going counterclockwise,
    #then rotates it 90 degrees counterclockwise to point outwards
    nPts=np.size(cShape)
    orientation=np.ones([nPts])*1j
    for ind in range(nPts):
        #get the direction then rotate by 90 degrees clockwise for it to point outward, since points are in counterclockwise order
        orientation[ind] = (((cShape[(ind)]) - (cShape[(ind-1) ])))*(-1j)
        orientation[ind] = orientation[ind]/abs(orientation[ind])
    if np.isnan(orientation).any():
        warnings.warn('Did not fill all orientation values')
    return orientation


def curveAngularPos(cShape):
    #get the center of mass than the unit vector pointing from here to each point
    x, y = get_center_boundary(np.real(cShape), np.imag(cShape))
    centerOfMass = x + y*1j
    angularPos=cShape-centerOfMass
    angularPos=angularPos/np.abs(angularPos)

    return angularPos


def make_natural_formlet(nPts=1000, radius=1, nFormlets=32, meanFormDir=-pi,
                       stdFormDir=pi/10, meanFormDist=1, stdFormDist=0.1,
                       startSigma=0.3, endSigma=0.1, randomstate = None):

    #set the seed for reproducibility
    if randomstate == None:
        randomstate = np.random.RandomState(np.random.rand(1))

#   make a circle
    angles = np.linspace(0.0, 2*pi, nPts)
    cShape = np.exp(angles*1j)*radius

    #where are the formlets centers going to be
    centers=np.ones(nPts)*1j
    for ind in range(nPts):
        #gaussian distributed with some bias towards a direction, but some jitter in distance from
        #origin and orientation
        centers[ind] = randomstate.normal(meanFormDist, stdFormDist)*np.exp(randomstate.normal(meanFormDir, stdFormDir)*1j)

    #what will be the scale of those formlets
    sigma = np.logspace(np.log10(startSigma), np.log10(endSigma), nFormlets)

    # roughly the sigma to alpha ratio random sign of gain
    alpha = 0.10*sigma*2*(randomstate.binomial(1, 0.5, nFormlets) - 0.5)

    #alpha = ((1.0/(-2.0*pi))*sigma)/1.1

    #apply formlet
    for ind in range(nFormlets):
        cShape = applyGaborFormlet(cShape, centers[ind], alpha[ind], sigma[ind])

    if cShape[0] is not cShape[-1]:
        cShape[-1] = cShape[0]

    x = np.real(cShape)
    y = np.imag(cShape)

    tck,u = interpolate.splprep([x, y], s=0, k=2)
    unew = np.linspace(0, 1, nPts)
    resample = np.array(interpolate.splev(unew, tck, der=0))
    cShape = np.sum(np.array([1,1j]) * resample.T, axis=1)

    return cShape, np.real(cShape), np.imag(cShape), sigma, alpha

def make_n_natural_formlets(**args):
    rng = np.random.RandomState(args['randseed'])
    s= []
    #I did this with **args so later on I could easily return them
    for ind in range(args['n']):
        cShape, x, y, sigma, alpha = make_natural_formlet(nPts=args['nPts'],
                                                    radius = args['radius'],
                                                    nFormlets = args['nFormlets'],
                                                    meanFormDir = args['meanFormDir'],
                                                    stdFormDir = args['stdFormDir'],
                                                    meanFormDist = args['meanFormDist'],
                                                    stdFormDist = args['stdFormDist'],
                                                    startSigma = args['startSigma'],
                                                    endSigma = args['endSigma'],
                                                    randomstate = rng)

        #resample and filter
        a_s = np.array([x, y]).T
        max_ext = np.max(np.abs(a_s))
        n_pix_per_side = args['min_n_pix']
        frac_of_image = args['frac_image']
        dists = np.sum((np.diff(a_s.T))**2, axis=0)**0.5
        scale = (n_pix_per_side*frac_of_image)/(max_ext*2.)
        dx = np.median(dists)*scale
        freqs = np.fft.fftfreq(len(x), dx)
        low_pass = np.zeros(np.shape(a_s))*1j
        low_pass[np.abs(freqs)<((0.5**0.5)/2.), :] = 1.
        ft = np.fft.fft(a_s, axis=0) * low_pass
        lp = np.real_if_close(np.fft.ifft(ft, axis=0))
#        lp = lp * scale
#        cx, cy = get_center_boundary(lp[:,1], lp[:,0])
#        lp[:, 1] = lp[:, 1] + (n_pix_per_side/2.0 - cx)
#        lp[:, 0] = lp[:, 0] + (n_pix_per_side/2.0 - cy)

        s.append(lp)

    return s

def applyGaborFormlet(cShape, center, alpha, sigma):

   alphaBounds = [(1.0/(-2.0*pi))*sigma, 0.1956*sigma]

   r = np.abs(cShape-center)


   if alphaBounds[0]>alpha or alphaBounds[1]<alpha:
        print('alpha is outside of the bounds for which Jordan curves are guaranteed')
        warnings.warn('alpha is outside of the bounds for which Jordan curves are guaranteed')

   cShapeUnitVectors = (cShape - center)/r
   newcShape = center + cShapeUnitVectors * (r + alpha * np.exp((-r**2.0) / sigma**2.0) * np.sin((2.0 * pi * r) / sigma))

   return newcShape


