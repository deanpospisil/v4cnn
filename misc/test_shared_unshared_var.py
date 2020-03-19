# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:50:05 2017

@author: deanpospisil
"""

import numpy as np
t = np.linspace(0,2*np.pi)
p1 = np.cos(t + np.deg2rad(-20))
p2 = np.cos(t + np.deg2rad(15))
r = np.cos(t + np.deg2rad(0))

p1 = p1/np.linalg.norm(p1)
p2 = p2/np.linalg.norm(p2)
r = r/np.linalg.norm(r)

p1r = np.dot(p1, r) * p1
p2r = np.dot(p2, r) * p2

e1 = r - p1r
e2 = r - p2r

e1v = np.sum(e1**2)
e2v = np.sum(e2**2)

p1rv = np.sum(p1r**2)
p2rv = np.sum(p2r**2)

p1e2v = np.dot(p1, e2)
p2e1v = np.dot(p2, e1)


#%%
usv1 = p1e2v
usv2 = p2e1v

s1v = p1rv - usv1
s2v = p2rv - usv2

print(s1v)
print(s2v)

print(usv1)
print(usv2)



