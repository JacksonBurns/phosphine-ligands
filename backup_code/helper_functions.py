"""
@uthor: Himaghna 15th Octobr 2018
Description: toolbox of helper functions
"""


from typing import List

import os
import glob
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt



class IterateSubdirectories(object):
    """
    Container object to iterate over all Sub-directories of a parent directory
    """
    def __init__(self, parent_directory):
        self.parent_directory = parent_directory

    def __iter__(self):
        for directory in (f.path for f in os.scandir(self.parent_directory)
                          if f.is_dir()):
            yield directory

class IterateFiles(object):
    """
    Container object to iterate over files with a given extension
    of a parent directory. In all files needed, extension = '*'
    """
    def __init__(self, parent_directory, extension):
        self.parent_directory = parent_directory
        self.extension = extension
        if not self.extension =='*':
            self.extension = '.' + self.extension

    def __iter__(self):
        for file in glob.glob(os.path.join(self.parent_directory,
                                           '*'+self.extension)):
            yield file


def load_pickle(file, dir=None):
    if dir is not None:
        fname = os.path.join(dir, file)
    else:
        fname = file
    X = pickle.load(open(fname, "rb"))
    return X


def plot_parity(x, y, **kwargs):
    plot_params = {
        'alpha': 0.7,
        's': 10,
        'c': 'green',
    }
    if kwargs is not None:
        plot_params.update(kwargs)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.scatter(x=x, y=y, alpha=plot_params['alpha'], s=plot_params['s'], c=plot_params['c'])
    max_entry = max(max(x), max(y)) + plot_params.get('offset', 5)
    min_entry = min(min(x), min(y))  - plot_params.get('offset', 5)
    axes = plt.gca()
    axes.set_xlim([min_entry, max_entry])
    axes.set_ylim([min_entry, max_entry])
    plt.plot([min_entry, max_entry], [min_entry, max_entry],
             color=plot_params.get('linecolor', 'black'))
    plt.title(plot_params.get('title', ''), fontsize=plot_params.get('title_fontsize', 24))
    plt.xlabel(plot_params.get('xlabel', ''),
                   fontsize=plot_params.get('xlabel_fontsize', 20))
    plt.ylabel(plot_params.get('ylabel', ''),
                   fontsize=plot_params.get('ylabel_fontsize', 20))
    plt.xticks(fontsize=plot_params.get('xticksize',24))
    plt.yticks(fontsize=plot_params.get('yticksize',24))
    if plot_params.get('show_plot', True):
        plt.show()
    return plt


