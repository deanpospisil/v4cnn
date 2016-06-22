# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:46:39 2015

@author: deanpospisil
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as  l

#the Nonlin fit model for Pasupathy V4 Neurons
mat = l.loadmat('V4_370PC2001_LSQnonlin.mat')
v4 = np.array(mat['fI'][0])[0]
v4 = v4[v4[:,-1]>0.5]
sns.set_context("talk", font_scale=1.4)

#the Nonlin fit model for Pasupathy V4 Neurons
layer = 5
mat = l.loadmat('AlexNet_370PC2001_LSQnonlin')
alex = np.array(mat['fI'][0])[layer]
alex = alex[ -np.isnan(alex[:,-1]) ]
alex = alex[alex[:,-1]>0.5]
toPlot = [v4,alex] 
#for s in apc[:]:
#    plt.scatter(s[:,0],s[:,1], color='r')
plt.close('all')

for lo in range(2):
    for ca in range(2):
        plt.figure(figsize=(4,8))
        for ind in range(2):
            plt.subplot(2,1,ind+1)
            
            if ca == 0:
                
                plt.scatter(360*( toPlot[ind][ :, 2]/(2.0*np.pi) ), 360*( toPlot[ind][ :, 0 ]/(2.0*np.pi) ), color='b', facecolors='none')
                plt.xlim((0,360))
                plt.ylim((0,360))
                plt.yticks([0,90, 180, 270, 360] )
                plt.xticks([0,90, 180, 270, 360] )
                
                if lo == 1:
                    plt.gca().set_xscale('log')
                    plt.xticks([ 1*10**-2 ,1,  1*10**2 ] )
                    plt.xlim( 1*10**-2, 1*10**3 )
                
                
            else:
                
                plt.scatter( toPlot[ind][ :, 3 ], toPlot[ind][ :, 1 ], color='b', facecolors='none')
                plt.xlim((0,1))
                plt.ylim((-1.01, 1.1))
                plt.yticks([-1, -0.5, 0, 0.5, 1] )
                plt.xticks([ 0, 0.5, 1] )
                
                
                if lo == 1:
                    plt.gca().set_xscale('log')
                    plt.xticks([0, 1*10**-4, 1*10**-2, 1*10**-0] )
                    plt.xlim((1*10**-4,1))
            
            if ind == 1:
                plt.title('AlexNet Layer 5')
            else:
                plt.title('V4')
                
                if ca == 0:
                    plt.ylabel('Mean Angular Position')
                    plt.xlabel('SD Angular Position')
                    
                else:
                    
                    plt.ylabel('Mean Curvature')
                    plt.xlabel('SD Curvature')
                
#            plt.tight_layout()
##    plt.gca().set_yscale('log')
##    plt.gca().set_xscale('log')
#    plt.xticks([1*10**-4, 1*10**-2, 1*10**-0 ,  1*10**2,   1*10**4] )
#    plt.yticks([1*10**-3,1*10**-2, 1*10**-1, 1*10**-0] )
#    plt.gca().xaxis.set_ticks_position('none') 
#    plt.gca().yaxis.set_ticks_position('none') 
#    #plt.gca().set_aspect('auto')
#
