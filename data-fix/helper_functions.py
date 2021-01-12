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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm  # NOQA


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

def validationPlot(x, y, x_valid, y_valid, labels=None, valid_labels=None, **kwargs):

    plt = plot_parity(x=x, y=y, labels=labels, xlabel='True Selectivity',ylabel='Predicted Selectivity',s=30,show_plot=False)

    plt.scatter(x=x_valid, y=y_valid, alpha=1, s=40, c='orange', edgecolors='black', label='Internal Validation')
    if(valid_labels is not None):
        i = 0;
        for ix,iy in zip(x_valid,y_valid):
            if(countGrossErrors(ix,iy)>0):
                color='red'
            else:
                color='black'
            plt.annotate(str(valid_labels[i]), # this is the text
                        (ix,iy), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,-15), # distance from text to points (x,y)
                        ha='center',  # horizontal alignment can be left, right or center
                        fontsize=15,
                        c=color)
            
            i = i + 1;

    plt.legend()
    plt.show()

def plot_parity(x, y,labels=None, **kwargs):
    plot_params = {
        'alpha': 0.7,
        's': 10,
        'c': 'green',
    }
    
    if kwargs is not None:
        plot_params.update(kwargs)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.scatter(x=x, y=y, alpha=plot_params['alpha'], s=plot_params['s'], c=plot_params['c'], label='Test Data')
    # max_entry = max(max(x), max(y)) + plot_params.get('offset', 5)
    # min_entry = min(min(x), min(y))  - plot_params.get('offset', 5)


    max_entry = 1
    min_entry = -1


    axes = plt.gca()
    axes.set_xlim([min_entry, max_entry])
    axes.set_ylim([min_entry, max_entry])
    plt.plot([min_entry, max_entry], [min_entry, max_entry],
             color=plot_params.get('linecolor', 'black'),label='Perfect Model')
    plt.title(plot_params.get('title', ''), fontsize=plot_params.get('title_fontsize', 24))
    plt.xlabel(plot_params.get('xlabel', ''),
                   fontsize=plot_params.get('xlabel_fontsize', 20))
    plt.ylabel(plot_params.get('ylabel', ''),
                   fontsize=plot_params.get('ylabel_fontsize', 20))
    plt.xticks(fontsize=plot_params.get('xticksize',24))
    plt.yticks(fontsize=plot_params.get('yticksize',24))

    plt.plot([0, 0],[-100, 100],color='black',linestyle='--',alpha=0.5)
    plt.plot([-100, 100],[0, 0],color='black',linestyle='--',alpha=0.5)

    if(labels is not None):
        i = 0;
        for ix,iy in zip(x,y):
            if(countGrossErrors(ix,iy)>0):
                color='red'
            else:
                color='black'
            plt.annotate(str(labels[i]), # this is the text
                        (ix,iy), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,-15), # distance from text to points (x,y)
                        ha='center',  # horizontal alignment can be left, right or center
                        fontsize=15,
                        c=color)
            
            i = i + 1;
    a = np.array([i[0] for i in x])
    b = np.array([i[0] for i in y])

    mod = sm.GLS(b, a)
    res = mod.fit()
    # print(res.summary())

    lm = LinearRegression(fit_intercept=True)
    lm.fit(a.reshape(-1, 1),b.reshape(-1, 1))
    message = '''
    Regression Results:
    Slope = {:.2f}
    Intercept = {:.2f} 
    R^2 = {:.2f}
    '''.format(
        lm.coef_[0][0],
        lm.intercept_[0],
        lm.score(a.reshape(-1, 1),b.reshape(-1, 1))
    )
    plt.annotate(message,(0.5,-0.5),ha='center',fontsize=15)
    temp = np.array([-100,100])
    plt.plot(temp, lm.coef_[0][0]*temp + lm.intercept_,linestyle='--',label='RBF-KPCA',c='blue')

    if plot_params.get('show_plot', True):
        plt.legend(fontsize=15)
        plt.show()
    return plt

def countGrossErrors(test,predict):
    GE = 0
    for (t,p) in zip(test,predict):
        if (t>0 and p<0) or (p>0 and t<0):
            GE = GE + 1
    return GE
