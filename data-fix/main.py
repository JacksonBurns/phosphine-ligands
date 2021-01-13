# utility libraries
import json, os, pickle, bisect, random, types, math
from copy import deepcopy as dc
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
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, RANSACRegressor, HuberRegressor
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
# helper functions defined elsewhere
from helper_functions import plot_parity, countGrossErrors, validationPlot
# wpca from https://github.com/jakevdp/wpca/
from wpca import WPCA
# model explaining
import shap
# scaffold splitter from chainer_chemistry
from scaffold_split import get_scaffold_idxs  # NOQA
# math
from scipy.stats import rankdata

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def loadData(zeroReplace=-1, fromXL=True, doSave=False, removeLessThan=None, removeGreaterThan=None):
    """
    Retrieves data from Excel file and optionally writes it out to serialized format
    """
    if(fromXL):
        configs = json.load(open(r'data_config.json'))
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

def loadValidationData(zeroReplace=-1, fromXL=True, doSave=False, removeLessThan=None):
    """
    Retrieves data from Excel file and optionally writes it out to serialized format
    """
    configs = json.load(open(r'validation_data_config.json'))
    xl_file = configs.get('xl_file')
    target_column = configs.get('target_column')
    descrptr_columns = configs.get('descrptr_columns')
    err_column = configs.get('error_column')
    name_col = configs.get('ligand_names')
    absolute_yield = configs.get('absolute_yield')

    df = pd.read_excel(xl_file)
    absYields = df[absolute_yield].to_numpy()
    
    # remove any data which does not have our yield cutoff
    if removeLessThan is not None:
        # print("Input data set ({} total): ".format(len(df.index)),df[name_col])
        remove_idxs = [i for i in range(0,len(absYields)) if absYields[i]<removeLessThan]
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
    
    return X, y, feature_weights, sample_weights, ligNames

def familySeparation(X,y,ligand_names,vectrs=[0,1,2]):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca_reduced = PCA()
    X_pca = pca_reduced.fit_transform(X_std)
    # plot 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # colors = ['red', 'green']
    family_column = [1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0]
    #plt.legend()
    # c = [colors[int(family)] for family in family_column] # conditional coloring
    c = [((i+1)/2,0,(i+1)/2) for i in y] # conditional coloring
    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.75, s=50)
    
    # for (x, y, z) in zip(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]]):
    #     ax.text(x,y,z,
    #         str(ligand_names.pop(0)),
    #         zdir=(1,1,1),
    #         fontsize=10)

    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    plt.show()
    print('Max PC1', y[np.argmax(X_pca[:, 0])])

def doWPCA(X,y,feature_weights,sample_weights,output=False,trainSize=0.8,varianceNeeded=0.95):
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # split response, features, and weights into train and test set
    randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test = train_test_split(X, y, feature_weights, sample_weights, train_size=trainSize, random_state=randState)
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
    # instantiate weighted PCA, train
    wpca = WPCA()
    wpca.fit(X_std_test,weights=f_weights_test)

    # find where adequate variance is explained
    cum_var = np.cumsum(wpca.explained_variance_ratio_)  # sum across
    n_eigenvectors_needed = bisect.bisect_left(cum_var, varianceNeeded) + 1  # fastest way according to: https://stackoverflow.com/questions/2236906/first-python-list-index-greater-than-x

    # do reduced wPCA
    wpca_reduced = WPCA(n_components=n_eigenvectors_needed)
    X_std_train = wpca_reduced.fit_transform(X_std_train, weights=f_weights_train)
    X_std_test = wpca_reduced.transform(X_std_test, weights=f_weights_test)

    # do regression
    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    
    if(output and countGrossErrors(y_test,y_predict)/len(y_predict)==0 and lm.score(X_std_train, y_std_train)>0.25 and MAE(y_true=y_test, y_pred=y_predict)<0.25):
        print('{} explained by {} eigen-vectors'.format(cum_var[n_eigenvectors_needed-1], n_eigenvectors_needed))
        print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
        print('R2 of training data: ', lm.score(X_std_train, y_std_train))
        print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))
        print('Random seed for tts: ', randState)
        plot_parity(x=y_test, y=y_predict, labels=None, xlabel='True Selectivity',ylabel='Predicted Selectivity')

    return lm.score(X_std_train, y_std_train),MAE(y_true=y_test, y_pred=y_predict),countGrossErrors(y_test,y_predict)/len(y_predict)

def doRidgeCV(X,y,feature_weights,sample_weights,output=False,trainSize=0.8):
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # split response, features, and weights into train and test set
    randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test = train_test_split(X, y, feature_weights, sample_weights, train_size=trainSize, random_state=randState)
    # scale features
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    # scale response
    y_std_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_std_test = y_scaler.transform(y_test.reshape(-1, 1))
    # pull variable for de-sclaing response
    y_sigma = y_scaler.scale_
    """
    Ridge (Tikhonov) CV - weighted
    """
    ridge = RidgeCV()
    ridge.fit(X_std_train, y_std_train.ravel(), sample_weight=s_weights_train.ravel())
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in ridge.predict(X_std_test)]
    if(output and countGrossErrors(y_test,y_predict)/len(y_predict)==0): # and ridge.score(X_std_train, y_std_train)>0.25 and MAE(y_true=y_test, y_pred=y_predict)<0.25):
        print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
        print('R2 of training data: ', ridge.score(X_std_train, y_std_train))
        print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))
        print('Random seed for tts: ', randState)
        plot_parity(x=y_test, y=y_predict, labels=None, xlabel='True Selectivity',ylabel='Predicted Selectivity')
    
    return ridge.score(X_std_train, y_std_train),MAE(y_true=y_test, y_pred=y_predict),countGrossErrors(y_test,y_predict)/len(y_predict)

def doLASSO(X,y,feature_weights,sample_weights,output=False,trainSize=0.8,cv=10):
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # split response, features, and weights into train and test set
    randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test = train_test_split(X, y, feature_weights, sample_weights, train_size=trainSize, random_state=randState)
    # scale features
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    # scale response
    y_std_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_std_test = y_scaler.transform(y_test.reshape(-1, 1))
    # pull variable for de-sclaing response
    y_sigma = y_scaler.scale_
    """
    LASSO - no weights
    """
    lasso = LassoCV(cv=cv,max_iter=100000)
    lasso.fit(X_std_train, y_std_train.ravel())
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lasso.predict(X_std_test)]
    if(output):# and countGrossErrors(y_test,y_predict)/len(y_predict)==0 and lasso.score(X_std_train, y_std_train)>0.25 and MAE(y_true=y_test, y_pred=y_predict)<0.25):
        print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
        print('R2 of training data: ', lasso.score(X_std_train, y_std_train))
        print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))
        print('Random seed for tts: ', randState)
        plot_parity(x=y_test, y=y_predict, labels=None, xlabel='True Selectivity',ylabel='Predicted Selectivity')
    return lasso.score(X_std_train, y_std_train),MAE(y_true=y_test, y_pred=y_predict),countGrossErrors(y_test,y_predict)/len(y_predict)

def doKPCASeparation(X,y,vectrs=[0,1,2]):

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=False)
    X_kpca = kpca.fit_transform(X_std)
    # plot 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = [((i+1)/2,0,(i+1)/2) for i in y] # conditional coloring
    ax.scatter(X_kpca[:, vectrs[0]], X_kpca[:, vectrs[1]], X_kpca[:, vectrs[2]], \
        c=c, alpha=0.75, s=50)
    
    # for (x, y, z) in zip(X_kpca[:, vectrs[0]], X_kpca[:, vectrs[1]], X_kpca[:, vectrs[2]]):
    #     ax.text(x,y,z,
    #         str(ligand_names.pop(0)),
    #         zdir=(1,1,1),
    #         fontsize=10)

    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    plt.show()
    print('Max PC1', y[np.argmax(X_kpca[:, 0])])

def doKPCA(X,y,ligand_names,sample_weights,output=False,trainSize=0.8,heatMap=False,gamma=None,randSeed=None, splitter=None):
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # split response, features, and weights into train and test set
    if randSeed is not None:
        randState = randSeed
    else:
        randState = random.randint(1,1e9)
    if splitter is None:
        X_train, X_test, y_train, y_test, s_weights_train, s_weights_test, ln_train, ln = train_test_split(X, y, sample_weights, ligand_names, train_size=trainSize, random_state=randState)
    elif splitter=='scaffold':
        # returns a list of indices for the two sets
        train_idxs, test_idxs = get_scaffold_idxs()

        # file containing all of the ligand names
        with open(r'ligands_names.txt','r') as file:
            # remove newlines
            names = [i.replace("\n","") for i in file.readlines()]
        
        # pull the names from the list of ligands only if they are still in the data set
        test_names = [names[i] for i in test_idxs if names[i] in ligand_names]
        train_names = [names[i] for i in train_idxs if names[i] in ligand_names]

        # pull the new indices
        train_idxs = [ligand_names.tolist().index(i) for i in train_names]
        test_idxs = [ligand_names.tolist().index(i) for i in test_names]

        # write the training and testing data
        X_train = np.array([X[i] for i in train_idxs])
        X_test = np.array([X[i] for i in test_idxs])
        y_train = np.array([y[i] for i in train_idxs])
        
        y_test = np.array([y[i] for i in test_idxs])
        # f_weights_train
        # f_weights_test = [X[i] for i in test_idxs]
        # s_weights_train
        # s_weights_test = [X[i] for i in test_idxs]
        ln_train = np.array([ligand_names[i] for i in train_idxs])
        ln = np.array([ligand_names[i] for i in test_idxs])
        
        print('~~~~~~~~~~~')
        print(ln_train.tolist())
        print('~~~~~~~~~~~')
    else:
        raise(NotImplementedError)
    # scale features
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    # scale response
    y_std_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_std_test = y_scaler.transform(y_test.reshape(-1, 1))
    # pull variable for de-sclaing response
    y_sigma = y_scaler.scale_ 
    """
    Radial Basis Function Kernel Principal Component Regression
    """
    # instantiate weighted PCA, train
    kpca = KernelPCA(kernel="rbf",n_components=4,gamma=gamma)
    X_std_train = kpca.fit_transform(X_std_train)
    X_std_test = kpca.transform(X_std_test)
    # explained_variance = np.var(X_std_train, axis=0)
    # explained_variance_ratio = explained_variance / np.sum(explained_variance)
    # print(np.cumsum(explained_variance_ratio))
    # print(s_weights_train.shape)

    # print("Training Set: ", ln_train)
    # print("Testing Set: ",ln)
    # print(len(ln_train)+len(ln))

    # do regression
    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train) #, sample_weight=s_weights_train.ravel())
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]

    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]

    if(output): # and r2_score(y_test,y_predict)>0.85): # countGrossErrors(y_test,y_predict)/len(y_predict)==0 and lm.score(X_std_train, y_std_train)>0.35 and MAE(y_true=y_test, y_pred=y_predict)<0.20):
        print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
        print('R2 of training data: ', lm.score(X_std_train, y_std_train))
        print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))
        print('Random seed for tts: ', randState)
        plot_parity(x=y_test, y=y_predict, labels=ln, xlabel='True Selectivity',ylabel='Predicted Selectivity',s=30)

    return lm.score(X_std_train, y_std_train),MAE(y_true=y_test, y_pred=y_predict),countGrossErrors(y_test,y_predict)/len(y_predict), r2_score(y_test,y_predict)

def doKMeansClusteringWithKPCA(X,y,sample_weights,vectrs=[0,1,2]):
    # instantiate scalers
    x_scaler = StandardScaler()
    # scale features
    X_scaled = x_scaler.fit_transform(X)
    """
    Radial Basis Function Kernel Principal Component Regression
    """
    kpca = KernelPCA(kernel="rbf",n_components=3)
    X_pca = kpca.fit_transform(X_scaled)
    """
    Cluster on the first 3 components 
    """
    # this plot shows that 4 clusters is optimal
    inertiaList = []
    for i in range(1,10):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(dc(X_pca), sample_weight=dc(sample_weights).ravel())
        inertiaList.append(kmeans.inertia_)
    plt.plot(range(1,10),[i for i in inertiaList])
    plt.ylabel("Inverse Inertia")
    plt.xlabel("Number of Clusters")
    plt.title("K-Means Clustering Elbow Plot")
    plt.show()
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(dc(X_pca), sample_weight=dc(sample_weights).ravel())

    # plot actual values and clustering side by side
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    c = [((i+1)/2,0,(i+1)/2) for i in y] # conditional coloring
    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.75, s=50)
    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    # clustering
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    colors = ['red','green','blue','orange']
    c = [colors[i] for i in kmeans.labels_] # conditional coloring
    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.75, s=50)
    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    for label in range(0,4):
        print('Cluster {} ({}) average and stdev:'.format(label,colors[label]))
        print(np.average(y[kmeans.labels_ == label]))
        print(np.std(y[kmeans.labels_ == label]))
        print('Max in this Cluster: ',np.max(y[kmeans.labels_ == label]))

    plt.show()

def doNonParametricValidation(ttd,vd,vectrs=[0,1,2]):
    # instantiate scalers
    x_scaler = StandardScaler()
    # scale features
    X_scaled = x_scaler.fit_transform(ttd.X)
    X_scaled_valid = x_scaler.transform(vd.X)
    """
    Radial Basis Function Kernel Principal Component Regression
    """
    kpca = KernelPCA(kernel="rbf",n_components=3)
    X_pca = kpca.fit_transform(X_scaled)
    X_pca_valid = kpca.transform(X_scaled_valid)
    """
    Cluster on the first 3 components 
    """
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(dc(X_pca), sample_weight=dc(ttd.s_weights).ravel())
    
    # plot actual values and clustering side by side
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    
    c = [(i,0,i) for i in (ttd.y-min(ttd.y))/(max(ttd.y)-min(ttd.y))] # conditional coloring

    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.75, s=50)
    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    # clustering
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    colors = ['red','green','blue','orange']
    c = [colors[i] for i in kmeans.labels_] # conditional coloring
    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.75, s=50)
    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)

    for label in range(0,kmeans.n_clusters):
        print('Cluster {} ({}) average and stdev:'.format(label,colors[label]))
        print(np.average(ttd.y[kmeans.labels_ == label]))
        print(np.std(ttd.y[kmeans.labels_ == label]))
        print('Max in this Cluster: ',np.max(ttd.y[kmeans.labels_ == label]))

    for i in range(0,len(vd.y)):
        coords = X_pca_valid[i,:]
        label = kmeans.predict(coords.reshape(1, -1))[0]
        ax.scatter(coords[0], coords[1], coords[2], c=colors[label], alpha=0.75, edgecolors='black',marker="*", s=350)
        ax.text(coords[0],coords[1],coords[2],str(vd.ln[i]),'x',
                        ha='right',  # horizontal alignment can be left, right or center
                        fontsize=15)
        print(vd.ln[i]," belongs to ",colors[label])

    plt.show()

def doKMCWithKPCAandSHAP(X,y,sample_weights,vectrs=[0,1,2],trainSize=0.8,randSeed=None):
    # instantiate scalers
    x_scaler = StandardScaler()
    # split response, features, and weights into train and test set
    if randSeed is not None:
        randState = randSeed
    else:
        randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, s_weights_train, s_weights_test = train_test_split(X, y, sample_weights, train_size=trainSize, random_state=randState)
    
    # scale features
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    """
    Radial Basis Function Kernel Principal Component Regression
    """
    kpca = KernelPCA(kernel="rbf",n_components=3)
    X_pca = kpca.fit_transform(X_std_train)
    X_pca_test = kpca.transform(X_std_test)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(dc(X_pca), sample_weight=dc(s_weights_train).ravel())

    # plot actual values and clustering side by side
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    c = [((i+1)/2,0,(i+1)/2) for i in y_train] # conditional coloring
    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.75, s=50)
    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    # clustering
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    colors = ['red','green','blue','orange']
    c = [colors[i] for i in kmeans.labels_] # conditional coloring
    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.75, s=50)
    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    for label in range(0,4):
        print('Cluster {} ({}) average and stdev:'.format(label,colors[label]))
        print(np.average(y_train[kmeans.labels_ == label]))
        print(np.std(y_train[kmeans.labels_ == label]))
        print('Max in this Cluster: ',np.max(y_train[kmeans.labels_ == label]))

    plt.show()

    explainer = shap.KernelExplainer(kmeans.predict,X_pca)
    shap_vals = explainer.shap_values(X_pca_test)
    shap.summary_plot(shap_vals,X_pca_test)

def doKMeansClustering(X,y,sample_weights,output=False,varianceNeeded=0.95):
    """
    K-Means Clustering
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_std, sample_weight=sample_weights.ravel())
    for label in range(0,3):
        print('Cluster {} average and stdev:'.format(label))
        print(np.average(y[kmeans.labels_ == label]))
        print(np.std(y[kmeans.labels_ == label]))
        print('Max in this Cluster: ',np.max(y[kmeans.labels_ == label]))

def doKPCATrainTestValid(ttd,vd,trainSize=0.8,randSeed=None):
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # split response, features, and weights into train and test set
    if randSeed is not None:
        randState = randSeed
    else:
        randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, s_weights_train, s_weights_test, ln_train, ln = train_test_split(ttd.X, ttd.y, ttd.s_weights, ttd.ln, train_size=trainSize, random_state=randState)
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

    """
    Radial Basis Function Kernel Principal Component Regression
    """
    # instantiate, reduce dimensionality
    kpca = KernelPCA(kernel="rbf",n_components=1)
    X_std_train = kpca.fit_transform(X_std_train)
    X_std_test = kpca.transform(X_std_test)
    # apply same transformation to validation data
    X_std_valid = kpca.transform(X_std_valid)

    # do regression
    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    # predict validation data, descale it
    y_valid_pred = [(i*y_sigma) + y_scaler.mean_ for i in lm.predict(X_std_valid)]
    y_valid_actual = [(i * y_sigma) + y_scaler.mean_ for i in y_std_valid]
    
    print('Training/Testing Data Statistics:')
    print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
    print('R2 of training data: ', lm.score(X_std_train, y_std_train))
    print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))

    print('Validation Data Statistics:')
    print('Mean Absolute Error: ', MAE(y_true=y_valid_actual,y_pred=y_valid_pred))
    print('% Gross Errors: ', countGrossErrors(y_valid_actual,y_valid_pred)/len(y_valid_pred))

    validationPlot(x=y_test, y=y_predict,x_valid=y_valid_actual,y_valid=y_valid_pred, labels=ln, valid_labels=vd.ln, xlabel='True Selectivity',ylabel='Predicted Selectivity',s=30)
    
    return

def visualizeBest(ttd, vd, randSeed=None, trainSize=0.8, gamma=None):
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # split response, features, and weights into train and test set
    if randSeed is not None:
        randState = randSeed
    else:
        randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, s_weights_train, s_weights_test, ln_train, ln = train_test_split(ttd.X, ttd.y, ttd.s_weights, ttd.ln, train_size=trainSize, random_state=randState)

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

    """
    Radial Basis Function Kernel Principal Component Regression
    """
    # instantiate, reduce dimensionality
    kpca = KernelPCA(kernel="rbf",n_components=1,gamma=0.05,random_state=randSeed)
    X_std_train = kpca.fit_transform(X_std_train)
    X_std_test = kpca.transform(X_std_test)
    # apply same transformation to validation data
    X_std_valid = kpca.transform(X_std_valid)
    # print(kpca.get_params())
    # print('default gamma: ',1/len(X_std_train[0,:]))
    # do regression
    
    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train*y_sigma + y_scaler.mean_, sample_weight=s_weights_train.ravel()**1.5)
    print(lm.score(X_std_train, y_std_train*y_sigma + y_scaler.mean_,sample_weight=s_weights_train**1.5))
    print(lm.coef_, " ", lm.intercept_)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    # predict validation data, descale it
    y_valid_pred = [(i*y_sigma) + y_scaler.mean_ for i in lm.predict(X_std_valid)]
    y_valid_actual = [(i * y_sigma) + y_scaler.mean_ for i in y_std_valid]
    
    '''
    # **1.5 weights
    # expfit = lambda x: -437.3*np.exp(0.2699*x)+437.6*np.exp(0.2661*x)
    # expfit = lambda x: -0.6984*(x-1.003)**3-0.7672
    # expfit = lambda x: -4.081*x**3+0.143
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in expfit(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    y_valid_pred = [(i*y_sigma) + y_scaler.mean_ for i in expfit(X_std_valid)]
    y_valid_actual = [(i * y_sigma) + y_scaler.mean_ for i in y_std_valid]
    '''
    # plot actual values and fitted line
    plt.figure()
    plt.subplot(1,2,2)
    c = [(i,0,i) for i in (y_std_train-min(y_std_train))/(max(y_std_train)-min(y_std_train))] # conditional coloring

    plt.scatter(X_std_train[:, 0], y_std_train*y_sigma + y_scaler.mean_, \
        c=c, alpha=0.75, s=rankdata(s_weights_train)**1.5)#=s_weights_train**1.5)
    plt.xlabel('PC 1', fontsize=20)
    plt.ylabel(f'NBMI', fontsize=20)
    plt.title("Training Data (Weighted)", fontsize=20)
    # preserve the auto-limits, as they are actually pretty good
    plt.xlim(plt.get('xlim'))
    plt.ylim(plt.get('ylim'))
    # plot the line the model is using
    temp = np.linspace(-1,1,100)
    plt.plot(temp, lm.coef_[0][0]*temp + lm.intercept_,linestyle='--',label='RBF-KPCA',c='blue')
    # plt.plot(temp, expfit(temp),linestyle='--',label='RBF-KPCA',c='blue')
    # print(s_weights_train**2)
    
    # plt.gca().invert_xaxis()

    plt.subplot(1,2,1)
    plt.scatter(X_std_train[:, 0], y_std_train*y_sigma + y_scaler.mean_, \
        c=c, alpha=0.75, s=50)
    plt.xlabel('PC 1', fontsize=20)
    plt.ylabel(f'NBMI', fontsize=20)
    plt.title("Training Data", fontsize=20)
    # preserve the auto-limits, as they are actually pretty good
    plt.xlim(plt.get('xlim'))
    plt.ylim(plt.get('ylim'))
    # plot the line the model is using
    temp = np.linspace(-1,1,100)
    plt.plot(temp, lm.coef_[0][0]*temp + lm.intercept_,linestyle='--',label='RBF-KPCA',c='blue')
    # plt.plot(temp, expfit(temp),linestyle='--',label='RBF-KPCA',c='blue')
    for ix,iy,i in zip(X_std_train[:, 0],y_std_train*y_sigma + y_scaler.mean_,range(len(X_std_train[:, 0]))):
        plt.annotate(str(ln_train[i]), (ix,iy), textcoords="offset points",xytext=(0,-15), # distance from text to points (x,y)
                            ha='center',  # horizontal alignment can be left, right or center
                            fontsize=10)
    plt.show(block=False)

    # print('x-data')
    # print(X_std_train.ravel(),X_std_test.ravel())
    # print('y-data')
    # print((y_std_train*y_sigma + y_scaler.mean_).ravel(),(y_std_test*y_sigma + y_scaler.mean_).ravel())
    # print('weights')
    # print((s_weights_train**1.5).ravel(),(s_weights_test**1.5).ravel())


    print('Training/Testing Data Statistics:')
    print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
    print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))

    print('Validation Data Statistics:')
    print('Mean Absolute Error: ', MAE(y_true=y_valid_actual,y_pred=y_valid_pred))
    print('% Gross Errors: ', countGrossErrors(y_valid_actual,y_valid_pred)/len(y_valid_pred))

    plt.figure()
    validationPlot(x=y_test, y=y_predict,x_valid=y_valid_actual,y_valid=y_valid_pred, labels=ln, valid_labels=vd.ln, xlabel='True Selectivity',ylabel='Predicted Selectivity',s=30)

def visualizeBest2D(ttd, vd, randSeed=None, trainSize=0.8, gamma=None):
    # instantiate scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # split response, features, and weights into train and test set
    if randSeed is not None:
        randState = randSeed
    else:
        randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, s_weights_train, s_weights_test, ln_train, ln = train_test_split(ttd.X, ttd.y, ttd.s_weights, ttd.ln, train_size=trainSize, random_state=randState)

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

    """
    Radial Basis Function Kernel Principal Component Regression
    """
    # instantiate, reduce dimensionality
    kpca = KernelPCA(kernel="rbf",n_components=2,gamma=0.05,random_state=randSeed)
    X_std_train = kpca.fit_transform(X_std_train)
    X_std_test = kpca.transform(X_std_test)
    # apply same transformation to validation data
    X_std_valid = kpca.transform(X_std_valid)
    # print(kpca.get_params())
    # print('default gamma: ',1/len(X_std_train[0,:]))
    # do regression
    
    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train*y_sigma + y_scaler.mean_, sample_weight=s_weights_train.ravel()**1.5)
    print(lm.score(X_std_train, y_std_train*y_sigma + y_scaler.mean_,sample_weight=s_weights_train**1.5))
    print(lm.coef_, " ", lm.intercept_)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    # predict validation data, descale it
    y_valid_pred = [(i*y_sigma) + y_scaler.mean_ for i in lm.predict(X_std_valid)]
    y_valid_actual = [(i * y_sigma) + y_scaler.mean_ for i in y_std_valid]
    
    # plot actual values and fitted line
    plt.figure()
    ax = plt.subplot(1,2,2,projection='3d')
    c = [(i,0,i) for i in (y_std_train-min(y_std_train))/(max(y_std_train)-min(y_std_train))] # conditional coloring

    ax.scatter(X_std_train[:, 0], X_std_train[:, 1], y_std_train*y_sigma + y_scaler.mean_, \
        c=c, alpha=0.75)#, s=50)#rankdata(s_weights_train)**1.5)
    ax.set_xlabel('PC 1', fontsize=20)
    ax.set_ylabel('PC 2', fontsize=20)
    ax.set_zlabel('NBMI', fontsize=20)
    ax.set_title("Training Data (Weighted)", fontsize=20)
    # preserve the auto-limits, as they are actually pretty good
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.set_zlim(ax.get_zlim())
    # plot the line the model is using
    temp = np.linspace(-0.4,0.6,100)
    ax.plot(temp, temp, lm.coef_[0][0]*temp + lm.coef_[0][1]*temp + lm.intercept_,linestyle='--',label='RBF-KPCA',c='blue')

    ax = plt.subplot(1,2,1,projection='3d')
    ax.scatter(X_std_train[:, 0], X_std_train[:, 1], y_std_train*y_sigma + y_scaler.mean_, \
        c=c, alpha=0.75)#, s=50)
    ax.set_xlabel('PC 1', fontsize=20)
    ax.set_ylabel('PC 2', fontsize=20)
    ax.set_zlabel('NBMI', fontsize=20)
    ax.set_title("Training Data", fontsize=20)
    # preserve the auto-limits, as they are actually pretty good
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.set_zlim(ax.get_zlim())
    # plot the line the model is using
    temp = np.linspace(-0.5,0.5,100)
    ax.plot(temp, temp, lm.coef_[0][0]*temp + lm.coef_[0][1]*temp + lm.intercept_,linestyle='--',label='RBF-KPCA',c='blue')
    
    # for ix,iy,i in zip(X_std_train[:, 0],y_std_train*y_sigma + y_scaler.mean_,range(len(X_std_train[:, 0]))):
    #     plt.annotate(str(ln_train[i]), (ix,iy), textcoords="offset points",xytext=(0,-15), # distance from text to points (x,y)
    #                         ha='center',  # horizontal alignment can be left, right or center
    #                         fontsize=10)
    plt.show()

    # print('x-data')
    # print(X_std_train.ravel())
    # print('y-data')
    # print((y_std_train*y_sigma + y_scaler.mean_).ravel())
    # print('weights')
    # print((s_weights_train**1.5).ravel())


    print('Training/Testing Data Statistics:')
    print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
    print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))

    print('Validation Data Statistics:')
    print('Mean Absolute Error: ', MAE(y_true=y_valid_actual,y_pred=y_valid_pred))
    print('% Gross Errors: ', countGrossErrors(y_valid_actual,y_valid_pred)/len(y_valid_pred))

    validationPlot(x=y_test, y=y_predict,x_valid=y_valid_actual,y_valid=y_valid_pred, labels=ln, valid_labels=vd.ln, xlabel='True Selectivity',ylabel='Predicted Selectivity',s=30)

def RBFKPCALogFit(ttd, vd, randSeed=None, trainSize=0.8, gamma=None):
    # instantiate scalers
    x_scaler = StandardScaler()

    # split response, features, and weights into train and test set
    if randSeed is not None:
        randState = randSeed
    else:
        randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, s_weights_train, s_weights_test, ln_train, ln = train_test_split(ttd.X, ttd.y, ttd.s_weights, ttd.ln, train_size=trainSize, random_state=randState)

    # scale features for train and test
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    # scale features for validation
    X_std_valid = x_scaler.transform(vd.X)

    """
    Radial Basis Function Kernel Principal Component Regression
    """
    # instantiate, reduce dimensionality
    kpca = KernelPCA(kernel="rbf",n_components=1,gamma=gamma,random_state=randSeed)
    X_std_train = kpca.fit_transform(X_std_train)
    X_std_test = kpca.transform(X_std_test)
    # apply same transformation to validation data
    X_std_valid = kpca.transform(X_std_valid)
    '''
    When plotted as NBMI vs. PC1, there is a somewhat obvious logarithmic distribution to the data.
    To fit a curve to this, first translate the data into the domain of the natural log and then 
    take the ln of all NBMI.
    '''
    tlt = 1.01
    y_train_logd = np.log(y_train+tlt)
    y_test_logd = np.log(y_test+tlt)
    y_valid_logd = np.log(vd.y+tlt)
    
    # do OLS regression
    lm = LinearRegression()
    lm.fit(X_std_train, y_train_logd, sample_weight=s_weights_train.ravel())
    print("Linear model R2: ",lm.score(X_std_train, y_train_logd, sample_weight=s_weights_train))
    # predict test data
    y_predict_logd = lm.predict(X_std_test)
    # predict validation data
    y_valid_pred_logd = lm.predict(X_std_valid)

    # do RANSAC regression
    ransac = RANSACRegressor()
    ransac.fit(X_std_train, y_train_logd)
    print("RANSAC R2: ",ransac.score(X_std_train, y_train_logd))
    # predict test data
    rs_y_predict_logd = ransac.predict(X_std_test)
    # predict validation data
    rs_y_valid_pred_logd = ransac.predict(X_std_valid)

    # do Huber regression
    huber = HuberRegressor()
    huber.fit(X_std_train, y_train_logd, sample_weight=s_weights_train.ravel())
    print("Huber R2: ",huber.score(X_std_train, y_train_logd, sample_weight=s_weights_train))
    # predict test data
    hb_y_predict_logd = huber.predict(X_std_test)
    # predict validation data
    hb_y_valid_pred_logd = huber.predict(X_std_valid)


    plt.figure()
    # plot actual values and fitted line
    plt.subplot(1,2,2)
    c = [(i,0,i) for i in (y_train-min(y_train))/(max(y_train)-min(y_train))] # conditional coloring
    plt.scatter(X_std_train[:, 0], y_train_logd, \
        c=c, alpha=0.75, s=rankdata(s_weights_train))
    plt.xlabel('PC 1', fontsize=20)
    plt.ylabel(f'ln(NBMI+{tlt})', fontsize=20)
    plt.title("Transformed Training Data (Weighted)", fontsize=20)
    # preserve the auto-limits, as they are actually pretty good
    plt.xlim(plt.get('xlim'))
    plt.ylim(plt.get('ylim'))
    # plot the line the model is using
    temp = np.linspace(-1,1,100)
    plt.plot(temp, lm.coef_[0]*temp + lm.intercept_,linestyle='--',label='OLS',c='blue')
    plt.plot(temp,ransac.predict(temp.reshape(-1, 1)),linestyle='--',label='RANSAC',c='red')
    plt.plot(temp,huber.predict(temp.reshape(-1, 1)),linestyle='--',label='Huber',c='green')
    plt.legend()
    # plot original data and show validation
    plt.subplot(1,2,1)
    plt.scatter(X_std_train[:, 0], y_train, \
        c=c, alpha=0.75, s=rankdata(s_weights_train))
    plt.xlabel('PC 1', fontsize=20)
    plt.ylabel(f'NBMI', fontsize=20)
    plt.title("Training Data (Weighted)", fontsize=20)
    # preserve the auto-limits, as they are actually pretty good
    plt.xlim(plt.get('xlim'))
    plt.ylim(plt.get('ylim'))
    plt.show(block=False)

    # untransform the model predictions
    y_predict = np.exp(y_predict_logd)-tlt
    y_valid_pred = np.exp(y_valid_pred_logd)-tlt

    print('Training/Testing Data Statistics:')
    print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
    print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))

    print('Validation Data Statistics:')
    print('Mean Absolute Error: ', MAE(y_true=vd.y,y_pred=y_valid_pred))
    print('% Gross Errors: ', countGrossErrors(vd.y,y_valid_pred)/len(y_valid_pred))

    plt.figure()
    validationPlot(x=y_test, y=y_predict,x_valid=vd.y,y_valid=y_valid_pred, labels=ln, valid_labels=vd.ln, xlabel='True Selectivity',ylabel='Predicted Selectivity',s=30)

def doRFR(ttd, vd, trainSize=0.8, randSeed=None):
    # split response, features, and weights into train and test set
    if randSeed is not None:
        randState = randSeed
    else:
        randState = random.randint(1,1e9)
    X_train, X_test, y_train, y_test, s_weights_train, s_weights_test, ln_train, ln = train_test_split(ttd.X, ttd.y, ttd.s_weights, ttd.ln, train_size=trainSize, random_state=randState)

    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(1,10,1),
            'n_estimators': range(1,50,1)
        },
        cv=5, scoring='neg_mean_absolute_error',# default RFR scoring
        verbose=1,n_jobs=-1,  # refit=True
        )
    # print(gsc.best_estimator_)
    # grid_result = gsc.fit(ttd.X, ttd.y, ttd.s_weights.ravel())
    grid_result = gsc.fit(X_train, y_train, s_weights_train.ravel())
    best_params = grid_result.best_params_
    print("best params: ",best_params, "MAE: ",-1*grid_result.best_score_)
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False, verbose=False)
    rfr.fit(X_train, y_train, s_weights_train.ravel())
    print("RFR R2: ",rfr.score(X_train, y_train, s_weights_train))
    y_predict = rfr.predict(X_test)
    y_valid_pred = rfr.predict(vd.X)
    validationPlot(x=y_test, y=y_predict,x_valid=vd.y,y_valid=y_valid_pred, labels=ln, valid_labels=vd.ln, xlabel='True Selectivity',ylabel='Predicted Selectivity',s=30)

if __name__ == '__main__':
    X, y, feature_weights, sample_weights, ligNames =  loadData(zeroReplace=0.01,removeLessThan=2,removeGreaterThan=None)
    ttd = types.SimpleNamespace(X=X, y=y, f_weights=feature_weights, s_weights=sample_weights, ln=ligNames)

    X, y, feature_weights, sample_weights, ligNames =  loadValidationData(zeroReplace=0.01,removeLessThan=-1)
    vd = types.SimpleNamespace(X=X, y=y, f_weights=feature_weights, s_weights=sample_weights, ln=ligNames)
    
    # doNonParametricValidation(ttd,vd)
    # doKPCATrainTestValid(ttd, vd, randSeed=837262349)

    doRFR(ttd,vd,randSeed=837262349)

    # visualizeBest(dc(ttd), dc(vd), randSeed=837262349, gamma=None, trainSize=0.8)
    # visualizeBest2D(dc(ttd), dc(vd), randSeed=837262349, gamma=0.02)
    # RBFKPCALogFit(dc(ttd), dc(vd), randSeed=None, gamma=None, trainSize=0.95)
    input()

    # fitR2s = []
    # for gamma in np.linspace(0.01,0.5,100):
    #     fitR2s.append(visualizeBest(dc(ttd), dc(vd), randSeed=837262349, gamma=gamma))
    # plt.plot(np.linspace(0.01,0.5,100),fitR2s)
    # plt.xlabel('gamma')
    # plt.ylabel('Fit R2')
    # plt.show()
    # familySeparation(dc(X),dc(y),dc(ligNames))
    # R2s=[]; MAEs=[]; GEs=[]; testR2s=[];
    # for i in range(0,10000):
        # R2, mae, GE = doWPCA(dc(X),dc(y),dc(feature_weights),dc(sample_weights),output=True)
        # R2, mae, GE = doRidgeCV(dc(X),dc(y),dc(feature_weights),dc(sample_weights),output=True)
        # R2, mae, GE = doLASSO(dc(X),dc(y),dc(feature_weights),dc(sample_weights),output=True)
    # R2, mae, GE, alsoR2 = doKPCA(dc(X),dc(y),dc(ligNames),dc(sample_weights),output=True,heatMap=False, splitter='scaffold' , randSeed=837262349)
        # R2s.append(R2); MAEs.append(mae); GEs.append(GE); testR2s.append(alsoR2);
    # doKPCASeparation(dc(X),dc(y))
    # doKMeansClusteringWithKPCA(dc(X),dc(y),dc(sample_weights))
    # doKMCWithKPCAandSHAP(dc(X),dc(y),dc(sample_weights),randSeed=454691077)
    # doKMeansClustering(dc(X),dc(y),dc(sample_weights))

    # plt.figure()
    # plt.hist(testR2s,rwidth=0.9,bins=[i/20 for i in range(0,20)],align='left')
    # plt.title('RBF-KPCA Distribution of Testing R^2 (1000 runs, 80:20 training:test)')
    # plt.ylabel('Frequency')
    # plt.xlabel('R^2')
    # plt.xticks([i/20 for i in range(0,20)],[str(i) for i in range(0,96,5)])
    # plt.show()
    # print("""Average %GE: {:.2f} Standard Deviation: {:.2f}""".format(np.average(GEs),np.std(GEs)))
    # print("""Average R2: {:.2f} Standard Deviation: {:.2f}""".format(np.average(R2s),np.std(R2s)))
    # print("""Average MAE: {:.2f} Standard Deviation: {:.2f}""".format(np.average(MAEs),np.std(MAEs)))
    # print("""Average Testing R2: {:.2f} Standard Deviation: {:.2f}""".format(np.average(testR2s),np.std(testR2s)))
