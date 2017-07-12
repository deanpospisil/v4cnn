#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:36:30 2017

@author: dean
"""

import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn')
sys.path.insert(0, top_dir + 'xarray/');
top_dir = top_dir + 'v4cnn/';
from matplotlib.backends.backend_pdf import PdfPages


pdf_files = top_dir+'analysis/figures/images/v4cnn_cur/'

#collect names of folder with figures to put in multipage pdf
figs = os.listdir(pdf_files)
figs = [fig for fig in figs if '.pdf' in fig]

fig_num = [int(fig.split('_')[0]) for fig in figs]
figs = [figs[i] for i in  np.argsort(fig_num)]
#create multipage pdf with blank pages
with PdfPages(pdf_files + 'multipage/multipage_short.pdf') as pdf:
    for ind in range(len(figs)):
        plt.figure(figsize=(8.5, 11))
        pdf.savefig()
        plt.close()

#%%
#copy those figures into our pdf 
blank_file = open(pdf_files + 'multipage/multipage_short.pdf', 'rb')
pdfwriter = PyPDF2.PdfFileWriter()

for i, fig in enumerate(figs):
    blank_pdf = PyPDF2.PdfFileReader(blank_file).getPage(i)
    loc_blank = blank_pdf.mediaBox.getUpperRight()
    fig_file = open(pdf_files + figs[i], 'rb')
    fig_pdf = PyPDF2.PdfFileReader(fig_file).getPage(0)
    loc_fig = fig_pdf.mediaBox.getUpperRight()
    
    blank_pdf.mergeTranslatedPage(fig_pdf, 
                                  tx=float(loc_blank[0])/2. - float(loc_fig[0])/2.,
                                  ty=float(loc_blank[1])/2. - float(loc_fig[1])/2.)

    pdfwriter.addPage(blank_pdf)

result = open(pdf_files +'multipage/merge_result.pdf', 'wb')
pdfwriter.write(result)
result.close()

