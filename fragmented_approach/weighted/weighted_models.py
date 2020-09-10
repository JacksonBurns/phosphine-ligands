"""
@uthor: Himaghna, 23rd September 2019
Description: Fit various models to predict E/Zs on the data-set
"""


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

import bisect
from wpca import WPCA

"""
Weighting the matrix is done in this way:
For a column of n experimental results (in this case isomeric ratio), apply the known experimental error 
to each results' descriptors (features) in an n by m array, where m is the number of features.

This is then provided in the call to WPCA as weights

"""


import data_processing
from helper_functions import plot_parity

# pyplot parameters
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)


def model(X,y,feature_weights, sample_weights,variance_needed=0.95,cv=10,train_size=0.80):
    """
    
    """
    out_maes = []
    out_err = []
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # split response, features, and weights into train and test set
    X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test = train_test_split(X, y, feature_weights, sample_weights, train_size=train_size)
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
    n_eigenvectors_needed = bisect.bisect_left(cum_var, variance_needed) + 1  # fastest way according to: https://stackoverflow.com/questions/2236906/first-python-list-index-greater-than-x
    # print('{} explained by {} eigen-vectors'.format(cum_var[n_eigenvectors_needed-1], n_eigenvectors_needed))

    # do reduced wPCA
    wpca_reduced = WPCA(n_components=n_eigenvectors_needed)
    X_std_train = wpca_reduced.fit_transform(X_std_train, weights=f_weights_train)
    X_std_test = wpca_reduced.transform(X_std_test, weights=f_weights_test)

    # do regression
    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    # print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    #     y_pred=y_predict))
    # print('R2 of training data: ', lm.score(X_std_train, y_std_train))
    # plot_parity(x=y_test, y=y_predict, xlabel='True Selectivity', \
    #     ylabel='Predicted Selectivity')
    out_maes.append(mean_absolute_error(y_true=y_test, y_pred=y_predict))
    out_err.append(count_terrible(y_test,y_predict))

    """
    LASSO - no weights
    """
    lasso = LassoCV(cv=cv,max_iter=100000)
    lasso.fit(X_std_train, y_std_train.ravel())
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lasso.predict(X_std_test)]
    # print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    #     y_pred=y_predict))
    # print('R2 of training data: ', lasso.score(X_std_train, y_std_train))
    # plt = plot_parity(x=y_test, y=y_predict, xlabel='True Selectivity', \
    #     ylabel='Predicted Selectivity')
    out_maes.append(mean_absolute_error(y_true=y_test, y_pred=y_predict))
    out_err.append(count_terrible(y_test,y_predict))

    """
    Ridge (Tikhonov) - weighted
    """
    ridge = Ridge()
    ridge.fit(X_std_train, y_std_train.ravel(), sample_weight=s_weights_train.ravel())
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in ridge.predict(X_std_test)]
    # print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    #     y_pred=y_predict))
    # print('R2 of training data: ', ridge.score(X_std_train, y_std_train))
    # plt = plot_parity(x=y_test, y=y_predict, xlabel='True Selectivity', \
    #     ylabel='Predicted Selectivity')
    out_maes.append(mean_absolute_error(y_true=y_test, y_pred=y_predict))
    out_err.append(count_terrible(y_test,y_predict))

    """
    Ridge (Tikhonov) CV - weighted
    """
    ridge = RidgeCV()
    ridge.fit(X_std_train, y_std_train.ravel(), sample_weight=s_weights_train.ravel())
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in ridge.predict(X_std_test)]
    # print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    #     y_pred=y_predict))
    # print('R2 of training data: ', ridge.score(X_std_train, y_std_train))
    # plt = plot_parity(x=y_test, y=y_predict, xlabel='True Selectivity', \
    #     ylabel='Predicted Selectivity')
    out_maes.append(mean_absolute_error(y_true=y_test, y_pred=y_predict))
    out_err.append(count_terrible(y_test,y_predict))
    return out_maes, out_err

def count_terrible(test,predict):
    terrible = 0
    for (t,p) in zip(test,predict):
        if (t>1 and p<1) or (p>1 and t<1):
            terrible = terrible + 1
    return terrible

def main():
    X = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\X.p', "rb"))
    y = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\y.p', "rb"))
    feature_weights = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\feature_weights.p', "rb"))
    sample_weights = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\sample_weights.p', "rb"))
    WPCAmaes = []; RidgeCVmaes = []; Ridgemaes = []; LASSOmaes = [];
    WPCAerr = []; RidgeCVerr = []; Ridgeerr = []; LASSOerr = [];
    for i in range(1000):
        if i%50==0:
            print(i)
        maes, err = model(X, y, feature_weights, sample_weights, variance_needed=0.95, train_size=0.80, cv=10)
        WPCAmaes.append(maes[0])
        LASSOmaes.append(maes[1])
        Ridgemaes.append(maes[2])
        RidgeCVmaes.append(maes[3])
        WPCAerr.append(maes[0])
        LASSOerr.append(maes[1])
        Ridgeerr.append(maes[2])
        RidgeCVerr.append(maes[3])

    F = 20;
    plt.figure()
    plt.hist([WPCAmaes, RidgeCVmaes],rwidth=0.9,bins=[i for i in range(0,F)],align='left',label=['WPCR','RidgeCV'])
    plt.legend(loc='best')
    plt.title('WPCA vs. RidgeCV Distribution of MAE (1000 runs, 80:20 training:test)')
    plt.ylabel('Frequency')
    plt.xlabel('MAE')
    plt.xticks([i for i in range(0,F)],[str(i) for i in range(0,F)])
    plt.show()

    plt.figure()
    plt.hist([WPCAerr, RidgeCVerr],rwidth=0.9,bins=[i for i in range(0,17)],align='left',label=['WPCA','RidgeCV'])
    plt.legend(loc='best')
    plt.title('WPCA vs. RidgeCV Distribution of Gross Errors out of 16 (1000 runs, 80:20 training:test)')
    plt.ylabel('Frequency')
    plt.xlabel('Count of Gross Errors')
    plt.xticks([i for i in range(0,17)],[str(i) for i in range(0,17)])
    plt.show()

    plt.figure()
    plt.hist([LASSOmaes, Ridgemaes],rwidth=0.9,bins=[i for i in range(0,F)],align='left',label=['LASSO','Ridge'])
    plt.legend(loc='best')
    plt.title('LASOO vs. Ridge Distribution of MAE (1000 runs, 80:20 training:test)')
    plt.ylabel('Frequency')
    plt.xlabel('MAE')
    plt.xticks([i for i in range(0,F)],[str(i) for i in range(0,F)])
    # plt.xticks([i/4 for i in range(0,16)],[str(round(i/4,2)) for i in range(0,16)])
    plt.show()

    plt.figure()
    plt.hist([LASSOerr, Ridgeerr],rwidth=0.9,bins=[i for i in range(0,17)],align='left',label=['LASSO','Ridge'])
    plt.legend(loc='best')
    plt.title('LASSO vs. Ridge Distribution of Gross Errors out of 16 (1000 runs, 80:20 training:test)')
    plt.ylabel('Frequency')
    plt.xlabel('Count of Gross Errors')
    plt.xticks([i for i in range(0,17)],[str(i) for i in range(0,17)])
    plt.show()

if __name__ == '__main__':
    main()
