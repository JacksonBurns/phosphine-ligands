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
import time

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


def model(X,y,feature_weights, sample_weights,ligand_names=None,variance_needed=0.95,cv=10,train_size=0.80):
    """
    
    """
    
    out_maes = []
    out_err = []
    outR2 = []
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # split response, features, and weights into train and test set
    X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test, l, ligand_test = train_test_split(X, y, feature_weights, sample_weights, ligand_names, train_size=train_size)
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

    print(X_std_train.size)
    print(y_std_train.size)
    # do regression
    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    
    # if(lm.score(X_std_train, y_std_train)>0.35 and count_terrible(y_test,y_predict)/len(y_predict)==0 and mean_absolute_error(y_true=y_test, y_pred=y_predict)<0.25):
    print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    y_pred=y_predict))
    print('R2 of training data: ', lm.score(X_std_train, y_std_train))
    
    print('training set:')
    for ii in l:
        print(ii)
    print('test set:')
    for ii in ligand_test:
        print(ii)
    plot_parity(x=y_test, y=y_predict, labels=ligand_test, xlabel='True Selectivity', \
        ylabel='Predicted Selectivity')
    outR2.append(lm.score(X_std_train, y_std_train))
    out_maes.append(mean_absolute_error(y_true=y_test, y_pred=y_predict))
    out_err.append(count_terrible(y_test,y_predict)/len(y_predict))

    # """
    # LASSO - no weights
    # """
    # lasso = LassoCV(cv=cv,max_iter=100000)
    # lasso.fit(X_std_train, y_std_train.ravel())
    # y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lasso.predict(X_std_test)]
    # # print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    # #     y_pred=y_predict))
    # # print('R2 of training data: ', lasso.score(X_std_train, y_std_train))
    # # plt = plot_parity(x=y_test, y=y_predict, xlabel='True Selectivity', \
    # #     ylabel='Predicted Selectivity')
    # outR2.append(lasso.score(X_std_train, y_std_train))
    # out_maes.append(mean_absolute_error(y_true=y_test, y_pred=y_predict))
    # out_err.append(count_terrible(y_test,y_predict)/len(y_predict))

    # """
    # Ridge (Tikhonov) - weighted
    # """
    # ridge = Ridge()
    # ridge.fit(X_std_train, y_std_train.ravel(), sample_weight=s_weights_train.ravel())
    # y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in ridge.predict(X_std_test)]
    # # print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    # #     y_pred=y_predict))
    # # print('R2 of training data: ', ridge.score(X_std_train, y_std_train))
    # # plt = plot_parity(x=y_test, y=y_predict, xlabel='True Selectivity', \
    # #     ylabel='Predicted Selectivity')
    # out_maes.append(mean_absolute_error(y_true=y_test, y_pred=y_predict))
    # outR2.append(ridge.score(X_std_train, y_std_train))
    # out_err.append(count_terrible(y_test,y_predict)/len(y_predict))

    # """
    # Ridge (Tikhonov) CV - weighted
    # """
    # ridge = RidgeCV()
    # ridge.fit(X_std_train, y_std_train.ravel(), sample_weight=s_weights_train.ravel())
    # y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in ridge.predict(X_std_test)]
    # # print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    # #     y_pred=y_predict))
    # # print('R2 of training data: ', ridge.score(X_std_train, y_std_train))
    # # plt = plot_parity(x=y_test, y=y_predict, xlabel='True Selectivity', \
    # #     ylabel='Predicted Selectivity')
    # out_maes.append(mean_absolute_error(y_true=y_test, y_pred=y_predict))
    # outR2.append(ridge.score(X_std_train, y_std_train))
    # out_err.append(count_terrible(y_test,y_predict)/len(y_predict))
    return out_maes, out_err, outR2

def count_terrible(test,predict):
    terrible = 0
    for (t,p) in zip(test,predict):
        if (t>0 and p<0) or (p>0 and t<0):
            terrible = terrible + 1
    return terrible

def main():
    start_time = time.time()
    X = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\X.p', "rb"))
    y = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\y.p', "rb"))
    feature_weights = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\feature_weights.p', "rb"))
    sample_weights = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\sample_weights.p', "rb"))
    ligand_names = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\ligand_names.p', "rb"))
    ligand_names = list(ligand_names)
    # feature_weights = feature_weights[:,0:X.shape[1]]
    # y = np.nan_to_num(y,nan=0.01)
    # sample_weights = np.nan_to_num(sample_weights,nan=0.01)
    # feature_weights = np.nan_to_num(feature_weights,nan=0.01)
    # X = np.nan_to_num(X,nan=0.01)
    WPCAmaes = []; RidgeCVmaes = []; Ridgemaes = []; LASSOmaes = [];
    WPCAerr = []; RidgeCVerr = []; Ridgeerr = []; LASSOerr = [];
    WPCAR2 = []; RidgeCVR2 = []; RidgeR2 = []; LASSOR2 = [];
    # print(X.size)
    for i in range(10):
        if (i+1)%1000==0:
            print("{} trials completed.".format(i+1))
        maes, err, R2 = model(X, y, feature_weights, sample_weights, ligand_names=ligand_names, variance_needed=0.95, train_size=0.80, cv=10)
        WPCAmaes.append(maes[0])
        # LASSOmaes.append(maes[1])
        #Ridgemaes.append(maes[2])
        #RidgeCVmaes.append(maes[3])
        WPCAerr.append(err[0])
        # LASSOerr.append(err[1])
        #Ridgeerr.append(err[2])
        #RidgeCVerr.append(err[3])
        WPCAR2.append(R2[0])
        # LASSOR2.append(R2[1])
        #RidgeR2.append(R2[2])
        #RidgeCVR2.append(R2[3]) 
    # print("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}; {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
    #     100*np.mean(LASSOR2),100*np.std(LASSOR2),np.mean(LASSOmaes),np.std(LASSOmaes),100*np.mean(LASSOerr),100*np.std(LASSOerr),
    #     100*np.mean(WPCAR2),100*np.std(WPCAR2),np.mean(WPCAmaes),np.std(WPCAmaes),100*np.mean(WPCAerr),100*np.std(WPCAerr)
    #     ))
    i = np.argmax(WPCAR2)
    print("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
        100*np.mean(WPCAR2),100*np.std(WPCAR2),np.mean(WPCAmaes),np.std(WPCAmaes),100*np.mean(WPCAerr),100*np.std(WPCAerr)
        ))
    print("Maximum R2 for WPCA: {:.2f} (%GE {:.2f}, MAE {:.2f})".format(WPCAR2[i],100*WPCAerr[i],WPCAmaes[i]))
    i = np.argmin(WPCAerr)
    print("Min %GE for WPCA: {:.2f} (R^2 {:.2f}, MAE {:.2f})".format(100*WPCAerr[i],WPCAR2[i],WPCAmaes[i]))
    i = np.argmin(WPCAmaes)
    print("Min MAE for WPCA: {:.2f} (R^2 {:.2f}, %GE {:.2f})".format(WPCAmaes[i],WPCAR2[i],100*WPCAerr[i]))

    # F = 20;
    # plt.figure()
    # plt.hist([WPCAmaes, RidgeCVmaes],rwidth=0.9,bins=[i for i in range(0,F)],align='left',label=['WPCR','RidgeCV'])
    # plt.legend(loc='best')
    # plt.title('WPCA vs. RidgeCV Distribution of MAE (1000 runs, 80:20 training:test)')
    # plt.ylabel('Frequency')
    # plt.xlabel('MAE')
    # plt.xticks([i for i in range(0,F)],[str(i) for i in range(0,F)])
    # plt.show()

    # plt.figure()
    # plt.hist([WPCAerr, RidgeCVerr],rwidth=0.9,bins=[i for i in range(0,17)],align='left',label=['WPCA','RidgeCV'])
    # plt.legend(loc='best')
    # plt.title('WPCA vs. RidgeCV Distribution of Gross Errors out of 16 (1000 runs, 80:20 training:test)')
    # plt.ylabel('Frequency')
    # plt.xlabel('Count of Gross Errors')
    # plt.xticks([i for i in range(0,17)],[str(i) for i in range(0,17)])
    # plt.show()

    # plt.figure()
    # plt.hist([LASSOmaes, Ridgemaes],rwidth=0.9,bins=[i for i in range(0,F)],align='left',label=['LASSO','Ridge'])
    # plt.legend(loc='best')
    # plt.title('LASOO vs. Ridge Distribution of MAE (1000 runs, 80:20 training:test)')
    # plt.ylabel('Frequency')
    # plt.xlabel('MAE')
    # plt.xticks([i for i in range(0,F)],[str(i) for i in range(0,F)])
    # # plt.xticks([i/4 for i in range(0,16)],[str(round(i/4,2)) for i in range(0,16)])
    # plt.show()

    # plt.figure()
    # plt.hist([LASSOerr, Ridgeerr],rwidth=0.9,bins=[i for i in range(0,17)],align='left',label=['LASSO','Ridge'])
    # plt.legend(loc='best')
    # plt.title('LASSO vs. Ridge Distribution of Gross Errors out of 16 (1000 runs, 80:20 training:test)')
    # plt.ylabel('Frequency')
    # plt.xlabel('Count of Gross Errors')
    # plt.xticks([i for i in range(0,17)],[str(i) for i in range(0,17)])
    # plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
