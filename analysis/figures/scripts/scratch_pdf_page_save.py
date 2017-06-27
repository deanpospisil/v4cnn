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

pdf_files = top_dir+'analysis/figures/images/v4cnn_cur/'
#create multipage pdf with blank pages
plt.figure(figsize=(8.5, 11))
plt.savefig(pdf_files + 'blank.pdf')
with PdfPages('multipage_pdf.pdf') as pdf:
    for ind in range()

    
#collect names of folder with figures to put in multipage pdf
figs = os.listdir(pdf_files)
figs = [fig for fig in figs if '.pdf' in fig]



#copy those figures into our pdf 
blank_file = open(pdf_files + 'blank.pdf', 'rb')
fig_file = open(pdf_files + figs[0], 'rb')
blank_pdf = PyPDF2.PdfFileReader(blank_file).getPage(0)
fig_pdf = PyPDF2.PdfFileReader(fig_file).getPage(0)
blank_pdf.mergeTranslatedPage(fig_pdf,tx=100,ty=300)
pdfwriter = PyPDF2.PdfFileWriter()
pdfwriter.addPage(blank_pdf)
fig_pdf.mediaBox.setLowerLeft((-100,-100))
fig_pdf.mediaBox.setUpperRight((612, 755))

pdfwriter.addPage(fig_pdf)


result = open(pdf_files+'merge_result.pdf', 'wb')
pdfwriter.write(result)
result.close()
blank_file.close()
fig_file.close()