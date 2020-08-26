"""
@uthor: Himaghna, 23rd September 2019
Description: Fit various models to predict E/Zs on the data-set
"""


import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
import time


import data_processing
from helper_functions import plot_parity

# processed_data = data_processing.main()
# X, y = processed_data['X'], processed_data['y'].reshape(-1,1)
# descriptor_names = processed_data['descriptor_names']
# # family_idx = processed_data['family_int']
# # # Mask Family
# # mask_id = 1
# # idx = []
# # for id, family_id in enumerate(family_idx):
# #     if family_id == mask_id:
# #         idx.append(id)
# # X, y = X[idx], y[idx] 
# pyplot parameters
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)


def do_PCR(X,y,descriptor_names,variance_needed=0.95):
    """
    Do PCR using top n_eigenvectors of the data matrix
    Params ::
    n_eigenvectors: int: Number of eigenvectors to retain. If set to None all
        are used. Defult None
    """
    # X = pickle.load(open(args.x, "rb"))
    # y = pickle.load(open(args.y, "rb"))
    # descriptor_names = pickle.load(open(args.descriptor_names, "rb"))

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    y_std_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_std_test = y_scaler.transform(y_test.reshape(-1, 1))
    y_sigma = y_scaler.scale_
    pca = PCA()
    pca.fit(X_std_test)
    def get_cumulative(in_list):
        """

        :param in_list: input list of floats
        :return: returns a cumulative values list
        """
        out_list = list()
        for key, value in enumerate(in_list):
            assert isinstance(value, int) or isinstance(value, float)
            try:
                new_cumulative = out_list[key-1] + value
            except IndexError:
                # first element
                new_cumulative = value
            out_list.append(new_cumulative)
        return out_list
    cum_var = get_cumulative([i for i in pca.explained_variance_ratio_])

    for key, value in enumerate(cum_var):
        if value >= variance_needed:
            n_eigenvectors_needed = (key+1)
            # print('{} explained by {} eigen-vectors'.format(value,
            #                                                 n_eigenvectors_needed))
            break

    # do reduced PCA
    pca_reduced = PCA(n_components=n_eigenvectors_needed)
    X_std_train = pca_reduced.fit_transform(X_std_train)
    X_std_test = pca_reduced.transform(X_std_test)

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
    return mean_absolute_error(y_true=y_test, \
        y_pred=y_predict), count_terrible(y_test,y_predict)



def do_LASSO(X,y,descriptor_names,cv=10):
    """
    Do LASSO on the data-set
    Params ::
    cv: int: folds of craoss-validation to do. Default 10
    Returns ::
    None
    """

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    y_std_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_std_test = y_scaler.transform(y_test.reshape(-1, 1))
    y_sigma = y_scaler.scale_


    lasso = LassoCV(cv=cv,max_iter=100000)
    lasso.fit(X_std_train, y_std_train.ravel())
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lasso.predict(X_std_test)]
    # print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
    #     y_pred=y_predict))
    # print('R2 of training data: ', lasso.score(X_std_train, y_std_train))

    # plt = plot_parity(x=y_test, y=y_predict, xlabel='True Selectivity', \
    #     ylabel='Predicted Selectivity')
    return mean_absolute_error(y_true=y_test, \
        y_pred=y_predict), count_terrible(y_test,y_predict)
    
    
def count_terrible(test,predict):
    terrible = 0
    for (t,p) in zip(test,predict):
        if (t>1 and p<1) or (p>1 and t<1):
            terrible = terrible + 1
    return terrible
    


def main():
    now = time.time()
    parser = ArgumentParser()
    parser.add_argument('-x', help='Path of X.p')
    parser.add_argument('-y', help='Path of y.p')
    parser.add_argument('-dn', '--descriptor_names', 
                        help='Path of descriptor_names.p')
    global args
    args = parser.parse_args()
    X = pickle.load(open(args.x, "rb"))
    y = pickle.load(open(args.y, "rb"))
    descriptor_names = pickle.load(open(args.descriptor_names, "rb"))
    pcr_maes = []
    pcr_gross_errors = []
    for i in range(0,1000):
        
        # X = pickle.load(open(args.x, "rb"))
        # y = pickle.load(open(args.y, "rb"))
        # descriptor_names = pickle.load(open(args.descriptor_names, "rb"))
        a, b = do_PCR(X,y,descriptor_names,variance_needed=0.95)
        pcr_maes.append(a)
        pcr_gross_errors.append(b)
    lasso_maes = []
    lasso_gross_errors = []
    for i in range(0,1000):
        if i%50 == 0:
            print(time.time() - now)
            print(str(i))
        # X = pickle.load(open(args.x, "rb"))
        # y = pickle.load(open(args.y, "rb"))
        # descriptor_names = pickle.load(open(args.descriptor_names, "rb"))
        a, b = do_LASSO(X,y,descriptor_names)
        lasso_maes.append(a)
        lasso_gross_errors.append(b)
    print(time.time() - now)
    plt.figure()
    plt.hist([pcr_maes, lasso_maes],rwidth=0.9,bins=[i for i in range(0,16)],align='left',label=['PCR','LASSO'])
    plt.legend(loc='best')
    plt.title('PCR vs. LASSO Distribution of MAE (1000 runs, 80:20 training:test)')
    plt.ylabel('Frequency')
    plt.xlabel('MAE')
    plt.xticks([i for i in range(0,16)],[str(i) for i in range(0,16)])
    plt.show()
    plt.figure()
    plt.hist([pcr_maes, lasso_maes],rwidth=0.9,bins=[i/4 for i in range(0,16)],align='left',label=['PCR','LASSO'])
    plt.legend(loc='best')
    plt.title('PCR vs. LASSO Distribution of MAE (1000 runs, 80:20 training:test)')
    plt.ylabel('Frequency')
    plt.xlabel('MAE')
    plt.xticks([i/4 for i in range(0,16)],[str(round(i/4,2)) for i in range(0,16)])
    plt.show()

    print(time.time() - now)
    # plt.figure()
    # plt.hist([pcr_gross_errors, lasso_gross_errors],rwidth=0.9,bins=[i for i in range(0,9)],align='left',label=['PCR','LASSO'])
    # plt.legend(loc='best')
    # plt.title('PCR vs. LASSO Distribution of Gross Errors (100 runs, 80:20 training:test)')
    # plt.ylabel('Frequency')
    # plt.xlabel('Count of Gross Errors')
    # plt.xticks([i for i in range(0,9)],[str(i) for i in range(0,9)])
    # plt.show()

if __name__ == '__main__':
    main()
