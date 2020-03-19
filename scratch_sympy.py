# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:52:56 2018

@author: deanpospisil
"""

from sympy import *
u = Symbol('u', real=True)
s = Symbol('s', real=True)

a = Symbol('a')
b = Symbol('b')
x = Symbol('x')
y = Symbol('y')
d = Symbol('d')
c = Symbol('c')

sol = solve([0.5*b + 0.5*b - u, 0.5*a + 0.5*b +(1/4)*(b-a)**2 - s], [a, b] ,set=True)
sol = solve([0.5*b + 0.5*b - u, 0.5*a + 0.5*b +(1/4)*(b-a)**2 - s], [a, b] ,set=True)

sol = solve([d+c*a-x, d+c*b-y], [d, c] ,set=True)

