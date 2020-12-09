import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
import time
import json

import bisect
from wpca import WPCA

import data_processing
from helper_functions import plot_parity

from prinpy.glob import NLPCA

# pyplot parameters
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)

def main():
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument('data_proc_config')
    args = parser.parse_args()

    configs = json.load(open(args.data_proc_config))
    xl_file = configs.get('xl_file')
    target_column = configs.get('target_column')
    descrptr_columns = configs.get('descrptr_columns')
    err_column = configs.get('error_column')
    out_dir = configs.get('output_directory')
    ligand_names = configs.get('ligand_names')

    df = pd.read_excel(xl_file)
    y = df[target_column].values
    ligands = df[ligand_names].values
    descriptor_names = descrptr_columns
    X = df[descrptr_columns].to_numpy()

    # pull and process weight column into shape of input array
    weights = df[err_column].to_numpy()  # read it
    weights = 1./weights  # invert
    weights = weights[np.newaxis]  # change from 1D to 2D
    weights = weights.T  # transpose to column vector
    sample_weights = weights
    feature_weights = np.repeat(weights, len(df[descrptr_columns].columns), axis=1)  # copy columns across to match input data

    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # split response, features, and weights into train and test set
    X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test, = train_test_split(X, y, feature_weights, sample_weights, train_size=0.8)
    # scale features
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    # scale response
    y_std_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_std_test = y_scaler.transform(y_test.reshape(-1, 1))
    # pull variable for de-sclaing response
    y_sigma = y_scaler.scale_

    """
    PCA - weighted
    """
    # instantiate weight PCA, train
    wpca = WPCA()
    wpca.fit(X_std_test,weights=f_weights_test)

    # find where adequate variance is explained
    cum_var = np.cumsum(wpca.explained_variance_ratio_)  # sum across
    n_eigenvectors_needed = bisect.bisect_left(cum_var, 0.95) + 1  # fastest way according to: https://stackoverflow.com/questions/2236906/first-python-list-index-greater-than-x
    # print('{} explained by {} eigen-vectors'.format(cum_var[n_eigenvectors_needed-1], n_eigenvectors_needed))

    # do reduced wPCA
    wpca_reduced = WPCA(n_components=n_eigenvectors_needed)
    X_std_train = wpca_reduced.fit_transform(X_std_train, weights=f_weights_train)
    X_std_test = wpca_reduced.transform(X_std_test, weights=f_weights_test)

    print(X_std_train.size)
    # NLPCA is the global alg
    # create solver
    pca = NLPCA()

    # transform data for better training with the 
    # neural net using built in preprocessor

    data_new = pca.preprocess([new_X, y])

    # fit the data
    pca.fit(data_new, epochs = 150, nodes = 15, lr = .01, verbose = 0)

    # project the current data. This returns a projection
    # index for each point and points to plot the curve
    proj, curve_pts = pca.project(data_new)
    plt.scatter(data_new[:,0], 
            data_new[:,1], 
            s = 5, 
            c = proj.reshape(-1), 
            cmap = 'viridis')
    plt.plot(curve_pts[:,0], 
         curve_pts[:,1], 
         color = 'black',
         linewidth = '1.5',
         linestyle='--')

    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
