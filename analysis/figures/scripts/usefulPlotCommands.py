# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:12:47 2015

@author: dean
"""

plt.xticks([0,90,180, 270, 360])
  
plt.xlim((0,360))
plt.ylim((-1,1.1))
plt.xlabel('Orientation (degrees)')
plt.ylabel('Normalized Curvature')
plt.rcParams.update({'font.size': 20})
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.imshow(img, cmap= plt.cm.Greys_r, interpolation = 'none')
fig, ax = plt.subplots(figsize=(12,6))
plt.gca().set_xticklabels(['0', .25, .5, .75, 1])