# utility libraries
import json, os, pickle, bisect, random, types, math, sys
from copy import deepcopy as dc
# math
from scipy.stats import rankdata
# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
from seaborn import heatmap
# data manipulation
import numpy as np
import pandas as pd
# sklearn regression tools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, RANSACRegressor, HuberRegressor, BayesianRidge
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
# wpca from https://github.com/jakevdp/wpca/
from wpca import WPCA
# helper functions defined elsewhere
from helper_functions import plot_parity, countGrossErrors, validationPlot, smallvalidationPlot
# model explaining
import shap
# data sampling
from dcekit.sampling import kennard_stone  # NOQA
from scaffold_split import get_scaffold_idxs  # NOQA

def loadData(config,zeroReplace=-1, fromXL=True, doSave=False, removeLessThan=None, removeGreaterThan=None):
    """
    Retrieves data from Excel file and optionally writes it out to serialized format
    """
    if(fromXL):
        configs = json.load(open(config))
        xl_file = configs.get('xl_file')
        target_column = configs.get('target_column')
        descrptr_columns = configs.get('descrptr_columns')
        err_column = configs.get('error_column')
        name_col = configs.get('ligand_names')
        absolute_yield = configs.get('absolute_yield')

        df = pd.read_excel(xl_file)
        absYields = df[absolute_yield].to_numpy()
        err_col = df[err_column].to_numpy()
        # remove any data which does not have our yield cutoff
        if removeLessThan is not None and removeGreaterThan is None:
            # print("Input data set ({} total): ".format(len(df.index)),df[name_col])
            remove_idxs = [i for i in range(0,len(absYields)) if absYields[i]<removeLessThan]
            print("Removing the following ligands ({} total):".format(len(remove_idxs)), ', '.join(df[name_col].to_numpy()[remove_idxs]))
            df.drop(remove_idxs,inplace=True)
            print("{} ligands remaining.".format(len(df.index)))
        elif removeGreaterThan is not None:
            remove_idxs = [i for i in range(0,len(err_col)) if err_col[i]>removeGreaterThan]
            print("Removing the following ligands ({} total):".format(len(remove_idxs)), ', '.join(df[name_col].to_numpy()[remove_idxs]))
            df.drop(remove_idxs,inplace=True)
            print("{} ligands remaining.".format(len(df.index)))

        ligNames = df[name_col].to_numpy()
        y = df[target_column].to_numpy()
        # replace any zeros to avoid inversion errors
        y[y==0] = zeroReplace
        descriptor_names = descrptr_columns
        X = df[descrptr_columns].to_numpy()
        # pull and process weight column into shape of input array
        weights = df[err_column].to_numpy()  # read it
        weights[weights == 0] = np.min(weights[weights > 0])  # replace 0 standard error with lowest other error
        weights = 1./weights  # invert
        weights = weights[np.newaxis]  # change from 1D to 2D
        weights = weights.T  # transpose to column vector
        sample_weights = weights
        feature_weights = np.repeat(weights, len(df[descrptr_columns].columns), axis=1)  # copy columns across to match input data
        del weights
    else:
        X = pickle.load(open(os.path.join(out_dir, 'X.p'), "rb"))
        y = pickle.load(open(os.path.join(out_dir, 'y.p'), "rb"))
        feature_weights = pickle.load(open(os.path.join(out_dir, 'feature_weights.p'), "rb"))
        sample_weights = pickle.load(open(os.path.join(out_dir, 'sample_weights.p'), "rb"))
        ligNames = pickle.load(open(os.path.join(out_dir, 'ligand_names.p'), "rb"))

    if(doSave):
        out_dir = configs.get('output_directory')
        X_path = os.path.join(out_dir, 'X.p')
        pickle.dump(X, open(X_path, "wb"))
        y_path = os.path.join(out_dir, 'y.p')
        pickle.dump(y, open(y_path, "wb"))
        descriptor_names_path = os.path.join(out_dir, 'descriptor_names.p')
        pickle.dump(descriptor_names, open(descriptor_names_path, "wb"))
        sample_weights_path = os.path.join(out_dir, 'sample_weights.p')
        pickle.dump(sample_weights, open(sample_weights_path, "wb"))
        feature_weights_path = os.path.join(out_dir, 'feature_weights.p')
        pickle.dump(feature_weights, open(feature_weights_path, "wb"))
        ligNames_path = os.path.join(out_dir, 'ligand_names.p')
        pickle.dump(ligNames, open(ligNames_path, "wb"))

    return X, y, feature_weights, sample_weights, ligNames

def splitData(ttd, randState=None, splitter=None, trainSize=0.80):
    # split response, features, and weights into train and test set
    if randState is None:
        randState = random.randint(1,1e9)
        print('Random Seed: ',randState)
    if splitter is None:
        X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test, ln_train, ln = train_test_split(ttd.X, ttd.y, ttd.f_weights, ttd.s_weights, ttd.ln, train_size=trainSize, random_state=randState)
    elif splitter == 'scaffold':
        # returns a list of indices for the two sets
        train_idxs, test_idxs = get_scaffold_idxs()
        # file containing all of the ligand names
        with open(r'ligands_names.txt','r') as file:
            # remove newlines
            names = [i.replace("\n","") for i in file.readlines()]
        # pull the names from the list of ligands only if they are still in the data set
        test_names = [names[i] for i in test_idxs if names[i] in ttd.ln]
        train_names = [names[i] for i in train_idxs if names[i] in ttd.ln]
        # pull the new indices
        train_idxs = [ttd.ln.tolist().index(i) for i in train_names]
        test_idxs = [ttd.ln.tolist().index(i) for i in test_names]
    elif splitter == 'kennard_stone':
        # get the appropriate number of training samples
        temp, _ = train_test_split(ttd.y, train_size=trainSize)
        train_idxs, test_idxs = kennard_stone(ttd.y.reshape(-1, 1),len(temp))
    else:
        raise(NotImplementedError)
    if splitter is not None:
        X_train = np.array([ttd.X[i] for i in train_idxs])
        X_test = np.array([ttd.X[i] for i in test_idxs])
        y_train = np.array([ttd.y[i] for i in train_idxs])
        y_test = np.array([ttd.y[i] for i in test_idxs])
        f_weights_train = np.array([ttd.f_weights[i] for i in train_idxs])
        f_weights_test = np.array([ttd.f_weights[i] for i in test_idxs])
        s_weights_train = np.array([ttd.s_weights[i] for i in train_idxs])
        s_weights_test = [ttd.s_weights[i] for i in test_idxs]
        ln_train = np.array([ttd.ln[i] for i in train_idxs])
        ln = np.array([ttd.ln[i] for i in test_idxs])
    return X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test, ln_train, ln

def doModel(model,center=0,splitter=None,randState=None, output=False, trainSize=0.8):
    # split the data into training and test
    X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test, ln_train, ln = splitData(ttd, randState=randState, splitter=splitter, trainSize=trainSize)

    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # scale features for train and test
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    # scale features for validation
    X_std_valid = x_scaler.transform(vd.X)
    # scale response for train and test
    y_std_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_std_test = y_scaler.transform(y_test.reshape(-1, 1))
    # scale response for validation
    y_std_valid = y_scaler.transform(vd.y.reshape(-1, 1))
    # pull variable for de-sclaing response
    y_sigma = y_scaler.scale_


    # fit the appropriate regressor
    # TODO: wrap these in an optional GridSearchCV
    if model == 'BayesianRidge':
        regressor = BayesianRidge()
    elif model == 'SVR':
        gsc = GridSearchCV(
            estimator=SVR(),
            param_grid={
                'C': [0.25,0.5,1,2,3,5,10],
                'epsilon': [0.05,0.1,0.2,0.5,0.75,1],
                'gamma': [0.025,0.05,0.1,0.2,0.5,1,2]
            }, cv=5, # scoring='neg_mean_absolute_error',
            verbose=0,n_jobs=-1,
        )
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        regressor = SVR(kernel='rbf',C=best_params["C"],epsilon=best_params["epsilon"],gamma=best_params["gamma"])
    elif model == 'KNN':
        gsc = GridSearchCV(
            estimator=KNeighborsRegressor(),
            param_grid={
                'n_neighbors': range(1,11),
                'weights': ['distance','uniform']
            }, cv=5, # scoring='neg_mean_absolute_error',
            verbose=0,n_jobs=-1,
        )
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        regressor = KNeighborsRegressor(n_neighbors=best_params["n_neighbors"], weights=best_params["weights"])
    elif model == 'RFR':
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'max_depth': range(1,10,1),
                'n_estimators': range(1,20,1)
            },
            cv=5, # scoring='neg_mean_absolute_error',
            verbose=0,n_jobs=-1,
        )
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        regressor = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False, verbose=False)
    elif model == 'kpca':
        # instantiate, reduce dimensionality
        kpca = KernelPCA(kernel="rbf",n_components=1,gamma=0.02)
        X_std_train = kpca.fit_transform(X_std_train)
        X_std_test = kpca.transform(X_std_test)
        # apply same transformation to validation data
        X_std_valid = kpca.transform(X_std_valid)
        # do regression
        regressor = LinearRegression()
    elif model == 'LASSO':
        gsc = GridSearchCV(
            estimator=LassoCV(),
            param_grid={
                'cv': range(1,6,1)
            }, cv=5, # scoring='neg_mean_absolute_error',
            verbose=0,n_jobs=-1,
        )
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        regressor = LassoCV(cv=best_params["cv"], max_iter=100000)
    elif model == 'RidgeCV':
        regressor = RidgeCV()
    elif model == 'WPCA':
        raise(NotImplementedError)
    else:
        raise(NotImplementedError)

    # TODO: add back option for weights
    regressor.fit(X_std_train, y_std_train)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in regressor.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    # predict validation data, descale it
    y_valid_pred = [(i*y_sigma) + y_scaler.mean_ for i in regressor.predict(X_std_valid)]
    y_valid_actual = [(i * y_sigma) + y_scaler.mean_ for i in y_std_valid]

    if(output):
        print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
        print(f'Results for {model} with {splitter}')
        print('Training/Testing Data Statistics:')
        print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
        print('R2 of training data: ', regressor.score(X_std_train, y_std_train))
        print('% Gross Errors: ', countGrossErrors(y_test,y_predict,c=center)/len(y_predict))

        print('Validation Data Statistics:')
        print('Mean Absolute Error: ', MAE(y_true=y_valid_actual,y_pred=y_valid_pred))
        print('% Gross Errors: ', countGrossErrors(y_valid_actual,y_valid_pred,c=center)/len(y_valid_pred))

    smallvalidationPlot(x=y_test, y=y_predict,x_valid=y_valid_actual,y_valid=y_valid_pred, labels=ln, valid_labels=vd.ln, xlabel='True Selectivity',ylabel='Predicted Selectivity',s=30)

    return


if __name__ == '__main__':
    X, y, feature_weights, sample_weights, ligNames =  loadData(r'data_config.json',zeroReplace=0.01,removeLessThan=0,removeGreaterThan=None)
    ttd = types.SimpleNamespace(X=X, y=y, f_weights=feature_weights, s_weights=sample_weights, ln=ligNames)

    X, y, feature_weights, sample_weights, ligNames =  loadData(r'validation_data_config.json',zeroReplace=0.01,removeLessThan=-1)
    vd = types.SimpleNamespace(X=X, y=y, f_weights=feature_weights, s_weights=sample_weights, ln=ligNames)

    plt.figure()
    plt.suptitle('Model and Sampling Screen for NBMI (abs. yield >2%, tts=0.70, zeros=0.01)',fontsize=12)
    models = ['BayesianRidge','SVR','KNN','RFR','kpca','LASSO','RidgeCV']
    splitters = ['scaffold','kennard_stone']
    for model in models:
        for splitter in splitters:
            plt.subplot(2, 7, models.index(model)+1+7*splitters.index(splitter))
            doModel(model,center=0,splitter=splitter,output=False,trainSize=0.70)
            plt.title(f'{model} with {splitter}',fontsize=8)

    plt.show()