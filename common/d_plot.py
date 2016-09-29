# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:43:37 2016

@author: dean
"""
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import d_img_process as imp


def tick_format_d(x, pos):
    if x==0:
        return('0')
    else:
        if x==1:
            return(str(x).split('.')[0])
        else:
            return(np.round(x, 2))


def nice_axes(axes, xticks=None, yticks=None, nxticks=5, nyticks=2):
    for i, an_axes in enumerate(axes):
        if i==len(axes)-1:
            if yticks==None:
                an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
                an_axes.set_yticks([])
            else:
                an_axes.set_yticks(yticks)
                an_axes.set_yticks([])
            if xticks==None:
               an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
            else:
                an_axes.set_xticks(xticks)
                an_axes.xaxis.set_tick_params(length=0)
                an_axes.yaxis.set_tick_params(length=0)
                an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
            an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
        else:
            an_axes.set_xticks([])
            an_axes.set_yticks([])

def stacked_hist_layers(cnn, logx=False, logy=False, xlim=None, maxlim=False,
                        bins=100, cumulative=False, normed=False,
                        extra_subplot=False, title=None):
    layers = cnn.index.get_level_values('layer_label').unique()
    if logx:
        cnn = np.log10(cnn.dropna())
        xlim = np.log10(xlim)
    if maxlim:
        xlim = [np.min(cnn.dropna().values), np.max(cnn.dropna().values)]

    n_subplot = len(layers)+extra_subplot
    for i, layer in enumerate(layers):
        plt.subplot(n_subplot, 1, i+1)

        if title!=None and i==0:
            plt.title(title)
        vals = cnn.loc[layer].dropna().values.flatten()
        plt.hist(vals, log=logy, bins=bins, histtype='step',
                 range=xlim, normed=normed, cumulative=cumulative)
        if cumulative:
            plt.ylim(0,1.1)
        plt.plot([np.median(vals),]*2, np.array(plt.gca().get_ylim()), color='red')
        plt.xlim(xlim)
        plt.gca().set_ylabel(layer, ha='right', rotation=0, labelpad=25)
        plt.gca().yaxis.set_label_position("right")
    if logx:
        plt.xlabel('log')
    nice_axes(plt.gcf().axes)

def vis_square(data, padsize=0, padval=0):
    plt.figure(figsize = (10,7.8))
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.xticks([])
    plt.yticks([])
    plt.imshow(data, interpolation='bicubic', cmap = cm.Greys_r)
    cbar=plt.colorbar(shrink=0.8)
    cbar.ax.set_ylabel('Normalized Firing Rate', rotation= 270, labelpad=15, fontsize = 15,)
    cbar.ax.yaxis.set_ticks([0,.25,.5,.75, 1])
    cbar.ax.set_yticklabels(['0', .25, .5, .75, 1])
    cbar.solids.set_rasterized(True)
    #plt.tight_layout()
    #plt.show()
    return data


def tick_format_d_int(x, pos):
    if x==0:
        return('0')
    else:
        return(str(round(x,0)).split('.')[0])


#def nice_axes(axes):
#    for i, an_axes in enumerate(axes):
#        an_axes.xaxis.set_tick_params(length=0)
#        an_axes.yaxis.set_tick_params(length=0)
#        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
#        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
def nice_axes_scatter_marg(axes, xticks=None, yticks=None, nxticks=2, nyticks=2):
    for i, an_axes in enumerate(axes):

        if yticks==None:
            an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
        else:
            an_axes.set_yticks(yticks)
        if xticks==None:
            an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
        else:
            an_axes.set_xticks(xticks)
        an_axes.xaxis.set_tick_params(length=0)
        an_axes.yaxis.set_tick_params(length=0)
        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d_int))
        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))

def find_count_unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx, u_counts = np.unique(b, return_index=True, return_counts=True)
    unique = a[idx]

    return unique, u_counts

def scatter_w_marginals(x, y, titlex, titley, xlim, ylim, xbins=None, ybins=None,
                        title=None):
    #first check if there is overlap in x, y
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1] )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[3])

    a = np.vstack((x,y)).T
    unique, counts = find_count_unique_rows(a)
    u_count = np.sort(np.unique(counts))

    if np.max(u_count)>1:#check there are in fact overlapping  points
        counts_s = (counts/np.double(max(counts))*100.)
        ax2.scatter(unique[:,0], unique[:,1], s=counts_s, alpha=0.7)

        u_count_s = np.sort(np.unique(counts_s))

        if len(u_count)>2:
            size = [np.min(u_count_s), np.median(u_count_s), np.max(u_count_s)]
            cnt = [np.min(u_count), int(np.median(u_count)), np.max(u_count)]
        else:
            size = [np.min(u_count_s),  np.max(u_count_s)]
            cnt = [np.min(u_count), np.max(u_count)]

        for_legend={'plt':[], 'label':[]}
        [for_legend['plt'].append(plt.scatter([],[], s=a_size)) for a_size in size]
        [for_legend['label'].append(str(a_cnt)) for a_cnt in cnt]

        # Put a legend to the right of the current axis
        ax2.legend(for_legend['plt'], for_legend['label'], frameon=True,
                         fontsize='medium', loc = 'top left', handletextpad=1,
                         title='Counts', scatterpoints = 1,fancybox=True,
                         framealpha=0.5, bbox_to_anchor=(-.15, -0.1))
    else:
        ax2.scatter(x, y)

    ax1.hist(y, orientation='horizontal', bins=ybins, range=ylim, align='mid')
    ax3.hist(x, orientation='vertical', bins=xbins, range=xlim, align='mid')

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_title(title)
    ax2.xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax2.yaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])

    ax1.xaxis.set_major_locator(mtick.LinearLocator(numticks=2, presets=None))
    ax1.yaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax1.yaxis.set_label_text(titley)

    ax3.xaxis.set_major_locator(mtick.LinearLocator(numticks=5, presets=None))
    ax3.yaxis.set_major_locator(mtick.LinearLocator(numticks=2, presets=None))
    ax3.xaxis.set_label_text(titlex)
    ax3.set_xlim(xlim)
    ax1.set_ylim(ylim)
    #nice_axes(fig.axes)
    #plt.show()
    return fig

def correct_bins_for_hist(bins):
    dif = np.diff(bins)
    bin_inds = list(range(len(bins))) + [len(bins)-1,]
    dif_inds = [0,] + list(range(len(dif))) + [len(dif)-1,]
    difs_to_add = dif[dif_inds]
    difs_to_add[:-1] = -difs_to_add[:-1]
    bins = bins[bin_inds]
    bins = bins + difs_to_add/2.
    return bins

def plot_resp_on_shapes(imgStack, resp, image_square = 19):
    resp_sc = (resp.values*0.8 +.2)
    imgStack = imgStack*resp_sc.reshape(362,1,1)
    #sort images
    sortStack = imgStack[list(reversed(np.argsort(resp_sc))),:,:]
    sortStack = np.array([imp.centeredCrop(img, 64, 64) for img in sortStack])
    data = vis_square(sortStack[0:image_square**2])
    #plt.title('Ranked response. ' + description, fontsize='x-large')
#    plt.tight_layout()
#    plt.show()
    return data
