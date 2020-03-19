# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:46:02 2017

@author: deanpospisil
"""

import itertools

pair = ['pair same texture', 'pair diff texture', 'pair same shape', 'pair diff shape', 'single stim']
dist = ['locally', 'globally']
ti = ['low ti', 'high ti']
sel = ['texture pref', 'form pref']
pool = ['max pool', 'lin pool', 'comp pool']

lists = [pair, dist, ti, sel, pool]
import itertools
all_possible_neurons = list(itertools.product(*lists))

text_file = open("Output.txt", "w")

for element in all_possible_neurons:
    print(element)
    text_file.write(str(element) + '\n')
    
text_file.close()
