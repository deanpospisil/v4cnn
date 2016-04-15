# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:53:09 2015

@author: dean
"""
import numpy as  np
import scipy.io as  l
import os
import matplotlib.pyplot as plt
import itertools
import scipy
plt.close('all')
fnum = [2, 5, 6, 11, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31, 
        33, 34, 37, 39, 43 ,44 ,45, 46, 48, 49, 50, 52, 54, 55, 56, 57, 58, 62, 
        66, 67, 68, 69, 70, 71 ,72, 74, 76, 77, 79, 80, 81, 83, 85, 86, 94, 104, 
        106, 108, 116, 117, 118, 123, 127,128 ,131, 133, 137, 138, 141, 142, 145, 
        152, 153, 154, 155, 156, 166, 170, 175, 190, 191, 193, 194]

maindir = '/Users/dean/Desktop/AlexNet_APC_Analysis/'
os.chdir( maindir)
resps = []

rxl = []
ryl = []

transPos = []
rfDiameter = []
for f in fnum:     
    mat = l.loadmat('PositionData_Yasmine/pos_'+ str(f)  +'.mat')
    
    rxl.append(np.squeeze(mat['data'][0][0][0]))
    ryl.append(np.squeeze(mat['data'][0][0][1]))    
    
    rx = np.double(np.squeeze(mat['data'][0][0][0]))
    ry = np.double(np.squeeze(mat['data'][0][0][1]))
    #print ry
    rfDiameter.append(np.sqrt( rx**2 + ry**2 )*0.625 + 40)    
    
    transPos.append(np.squeeze(mat['data'][0][0][2]))
    resps.append(np.squeeze(mat['data'][0][0][3]))

rxl = np.array(rxl)   
ryl = np.array(ryl) 
corrl = []
fracRFtrans = []
for trans, resp, cellInd in itertools.izip(transPos,resps, range(len(fnum))):
    
    stimPerPos = np.product(np.shape(resp[0]))
    numPos = len(trans) 
    stimResByPos = np.zeros( ( stimPerPos, numPos  ) )
    
    for posResp, pos in itertools.izip( resp, range(numPos)):
        stimResByPos[:,pos] = posResp.reshape(stimPerPos)
    centerCorr = np.corrcoef(stimResByPos.T)[trans==0].reshape(numPos)
    corrl.append(centerCorr)
    #fracRFtrans.append(trans / rfDiameter[cellInd])
    #fracRFtrans.append(trans / 1)
    fracRFtrans.append(trans / rfDiameter[cellInd]*2)
    
    plt.plot(fracRFtrans[cellInd], centerCorr, alpha = 0.2)
    plt.scatter(fracRFtrans[cellInd], centerCorr,alpha = 0.2)
#    
#lets get the mean response and correlation
corrPosPairs = []
for r, p in itertools.izip(corrl, fracRFtrans):
#    r = r - np.mean(r)
    for corr, pos in itertools.izip(r, p):   
        corrPosPairs.append([corr,pos])
corrPosPairs = np.array(corrPosPairs)    
sortPosInd=np.argsort(corrPosPairs[:,1])
corrPosPairs = corrPosPairs[sortPosInd,:]


#lets get the mean response and correlation, for eccentric
corrPosPairsEcc = []
for r, p in itertools.izip(corrl, fracRFtrans):
    r = r[p<0]
    r = r - np.mean(r)
    for corr, pos in itertools.izip(r, p):   
        corrPosPairsEcc.append([corr,pos])
corrPosPairsEcc = np.array(corrPosPairsEcc)    
sortPosInd=np.argsort(corrPosPairsEcc[:,1])
corrPosPairsEcc = corrPosPairsEcc[sortPosInd,:]

width = 40
mpos = []
mcor = []
for ind in range(len(corrPosPairs)-width):
    mcor.append(np.mean(corrPosPairs[ind:ind+width, 0]))
    mpos.append(np.mean(corrPosPairs[ind:ind+width, 1]))
#plt.figure() 
#plt.scatter(mpos,mcor) 

#
#mpos = []
#mcor = []
#mcsd = []
#width=0.15
#for dist in np.arange(np.min(corrPosPairs[:,1]), np.max(corrPosPairs[:,1]),0.01):
#    window = (dist<corrPosPairs[:,1]) * (corrPosPairs[:,1]<width+dist)
#    if not any(corrPosPairs[window,1] == 0) or np.sum(window)<20:
#        mcor.append(np.mean(corrPosPairs[window,0]))
#        mpos.append(np.mean(corrPosPairs[window,1]))
#        mcsd.append(np.std(corrPosPairs[window,0]))
#    else:
#        mcor.append(np.nan)
#        mpos.append(np.nan)
#        mcsd.append(np.nan)
#    
#mcsd = np.array(mcsd)
#mpos = np.array(mpos)
#mcor = np.array(mcor)
#plt.plot(mpos,mcor, linewidth = 4) 
#plt.fill_between(mpos, mcor+mcsd, mcor-mcsd, alpha=0.5)


pos = corrPosPairs[corrPosPairs[:,1]<0,1]
cor = corrPosPairs[corrPosPairs[:,1]<0,0]
regression = np.polyfit(pos, cor, 1)

regline = pos*regression[0]+regression[1]

r = np.corrcoef([regline, cor ])
plt.plot(pos, regline)

pos = corrPosPairsEcc[:,1]
cor = corrPosPairsEcc[:,0]
regression = np.polyfit(pos, cor, 1)
plt.scatter(pos,cor, facecolor = 'none')
regline = pos*regression[0]+regression[1]
plt.plot(pos,regline, color = 'b')
r = np.corrcoef([regline, cor ])
plt.text(pos[-1]+0.01,regline[-1], 'r = '+ str(round(r[0,1],3)) +', b = ' +str(round(regression[0],3))  )

#
#cell = 60
#resp = resps[cell]
#trans = transPos[cell]
#stimPerPos = np.product(np.shape(resp[0]))
#numPos = len(trans)
#fsize = 15
#
#stimResByPos = np.zeros( ( stimPerPos, numPos  ) )
#for posResp, pos in itertools.izip( resp, range(numPos)):
#    stimResByPos[:,pos] = posResp.reshape(stimPerPos)
#
#print rfDiameter[cell]
#
#fig = plt.figure(figsize=(12,4))
#plt.subplot(131)
##plt.scatter( np.tile(trans, (stimPerPos,1)), stimResByPos)
#mr = np.mean(stimResByPos,0)
#nmr = mr / np.max(mr)
#
#plt.stem( trans,nmr,color= 'r')
#plt.xlim(-rfDiameter[cell]/2., rfDiameter[cell]/2.)
#plt.ylim(0,1.1)
#
#plt.title('V4 Cell ' + str(fnum[cell]) )
#plt.xlabel(' Degrees Visual Angle from RF center')
#plt.ylabel('Normalized Mean Response',fontsize = fsize)
#plt.yticks([0, 0.25, 0.5, 0.75, 1]) 
#labels = [item.get_text() for item in plt.gca().get_yticklabels()]
##
#labels[0] = '0'
#labels[1] = '0.25'
#labels[2] = '0.5'
#labels[3] = '0.75'
#labels[-1] = '1'
#plt.gca().set_yticklabels(labels)
#plt.tick_params(axis='x',  which='both',bottom='on', top='off',labelbottom='on')
#
#
#plt.subplot(132)
#centerCorr = np.corrcoef(stimResByPos.T)[trans==0].reshape(numPos)
#plt.stem(trans, centerCorr)
#
#plt.ylim(0,1.1)
#plt.xlim(-rfDiameter[cell]/2., rfDiameter[cell]/2.)
#
#plt.ylabel('Correlation', fontsize = fsize)
#    
##plt.gca().set_aspect('equal', adjustable='box') 
#    
#plt.yticks([0, 0.25, 0.5, 0.75, 1]) 
#labels = [item.get_text() for item in plt.gca().get_yticklabels()]
##
#labels[0] = '0'
#labels[1] = '0.25'
#labels[2] = '0.5'
#labels[3] = '0.75'
#labels[-1] = '1'
##
#plt.gca().set_yticklabels(labels)
#
#plt.tick_params(axis='x',  which='both',bottom='on', top='off',labelbottom='on')
#plt.tight_layout()
#
#plt.subplot(133)
#respPos0 = stimResByPos[:,trans==0]
#respPos1 = stimResByPos[:,trans>0]
#plt.scatter(respPos0, respPos1 , facecolors='none'  )
#
#plt.xticks([0, round(np.nanmax(respPos0))])
#plt.yticks([0, round(np.nanmax(respPos1))])
#
#plt.gca().set_aspect('equal', adjustable='box')
#plt.xlabel('x= ' + str(trans[trans==0]),fontsize = fsize)
#plt.ylabel('x= ' + str(trans[trans>0]),fontsize = fsize)
