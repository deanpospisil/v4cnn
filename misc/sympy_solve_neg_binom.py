#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:56:49 2018

@author: dean
"""

from sympy import *

n = Symbol('n')
p = Symbol('p')
mu = Symbol('mu')
s = Symbol('s')

sol =solve([Eq((n*(1-p))/p-mu), Eq((n*(1-p))/(p**2)-s)],[n,p])
print(sol)
