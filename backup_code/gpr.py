"""
@uthor: Himaghna, 31st October 2019
Description: Perform GPR on the molecules to predict E/Z
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import data_processing
from data_post_process import bin_to_families, get_family_membership_idx, \
    get_train_test_idx
from helper_functions import plot_parity

from argparse import ArgumentParser
import pickle


# RC PArams
plt.rcParams['svg.fonttype'] = 'none'
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
processed_data = data_processing.main(r'data_proc_config.json')
X, y = processed_data['X'], processed_data['y'].reshape(-1,1)
# descriptor_names = processed_data['descriptor_names']
# family_idx = processed_data['family_int']


def process_X_y(training_size, random_seed, stratified=False):
    """
    Split X and y into training and testing set
    Params ::
    training_size: float (0., 1.]: size of the training set
    random_seed: int: random seed for the split
    stratified: Boolean: if set to True, the split samples from each strata/ bin
        /family of response. Default is False
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    if float(training_size) == 1.:
        # use entire set as training set
        X_train = x_scaler.fit_transform(X)
        X_test = X_train
        y_train = y_scaler.fit_transform(y)
        y_test = y_train
    else:
        if stratified is True:
            strata_bounds = [5]
            _, strata_bins = bin_to_families(y, bin_upper_bounds=strata_bounds)
            strata_dicts = get_family_membership_idx(strata_bins)
            train_test_idx = get_train_test_idx(strata_dicts, \
                train_size=training_size, random_state=random_seed)
            X_train, y_train = X[train_test_idx['train']], \
                y[train_test_idx['train']]
            X_test, y_test = X[train_test_idx['test']], \
                y[train_test_idx['test']]

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, \
                train_size=training_size, random_state=random_seed)
        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train)
        X_test = x_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_scaler': y_scaler,
        'x_scaler': x_scaler}



def do_gpr(**kwargs):
    training_size = kwargs.get('training_size', 0.80)
    stratified = kwargs.get('stratified', False)
    random_seed = kwargs.get('random_seed', 12345)

    # Get training and test data
    processed_input = process_X_y(training_size,
                                  random_seed,
                                  stratified=stratified)
    X_train = processed_input['X_train']
    X_test = processed_input['X_test']
    x_scaler = processed_input['x_scaler']
    y_train = processed_input['y_train']
    y_test = processed_input['y_test']
    y_scaler = processed_input['y_scaler']

    kernel = kwargs.get('kernel', RBF())
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X_train, y_train)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, pred_sigma = gpr.predict(X_test, return_std=True)
    # Scale up y
    y_pred_scaled = [_ * float(y_scaler.scale_) for _ in y_pred.flatten()]
    y_test_scaled = [_ * float(y_scaler.scale_) for _ in y_test.flatten()]
    pred_sigma_scaled = [_ * float(y_scaler.scale_) for _ in pred_sigma.flatten()]
    # define 95% confidence intervals
    ci_upper = [y_pred_val + 1.* sigma_val for \
        y_pred_val, sigma_val in zip(y_pred_scaled, pred_sigma_scaled)]
    ci_lower = [y_pred_val - 1. * sigma_val for \
        y_pred_val, sigma_val in zip(y_pred_scaled, pred_sigma_scaled)]
    # MAE
    print(mean_absolute_error(y_true=y_test_scaled, y_pred=y_pred_scaled))

    test_idx = [_ for _ in range(len(y_pred_scaled))]
    plt.scatter(x= test_idx, y=y_pred_scaled, c='red', marker='o', label='Prediction')
    plt.scatter(x=test_idx, y=y_test_scaled, c='black', marker='x', label='True Values')
    plt.fill_between(x=test_idx, y1=ci_upper, y2=ci_lower, alpha=0.4, color='blue', label='One SDV.')
    plt.xlabel('Training Example', fontsize=20)
    plt.ylabel('Selectivity Prediction', fontsize=20)
    plt.legend()
    plt.show()

def main():
    # parser = ArgumentParser()
    # parser.add_argument('-x', help='Path of X.p')
    # parser.add_argument('-y', help='Path of y.p')
    # parser.add_argument('-dn', '--descriptor_names', 
    #                     help='Path of descriptor_names.p')
    # args = parser.parse_args()
    # global X, y, descriptor_names
    # X = pickle.load(open(args.x, "rb"))
    # y = pickle.load(open(args.y, "rb"))
    # descriptor_names = pickle.load(open(args.descriptor_names, "rb"))
    do_gpr(random_seed=4200, training_size=0.99, stratified=True)


if __name__ == '__main__':
    main()



