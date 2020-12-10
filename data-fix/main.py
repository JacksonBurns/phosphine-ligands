# utility libraries
import json, os, pickle, bisect, random
from copy import deepcopy as dc
# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
# data manipulation
import numpy as np
import pandas as pd
# sklearn regression tools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
# helper functions defined elsewhere
from helper_functions import plot_parity, countGrossErrors
# wpca from https://github.com/jakevdp/wpca/
from wpca import WPCA

def loadData(zeroReplace=-1, fromXL=True, doSave=False):
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

        df = pd.read_excel(xl_file)
        ligNames = df[name_col].to_numpy()
        y = df[target_column].to_numpy()
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
    if(output and countGrossErrors(y_test,y_predict)/len(y_predict)==0 and ridge.score(X_std_train, y_std_train)>0.25 and MAE(y_true=y_test, y_pred=y_predict)<0.25):
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
    if(output and countGrossErrors(y_test,y_predict)/len(y_predict)==0 and lasso.score(X_std_train, y_std_train)>0.25 and MAE(y_true=y_test, y_pred=y_predict)<0.25):
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

def doKPCA(X,y,output=False,trainSize=0.8):
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
    Radial Basis Function Kernel Principal Component Regression
    """
    # instantiate weighted PCA, train
    kpca = KernelPCA(kernel="rbf",n_components=3)
    X_std_train = kpca.fit_transform(X_std_train)
    X_std_test = kpca.transform(X_std_test)

    # do regression
    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    
    if(output and countGrossErrors(y_test,y_predict)/len(y_predict)==0):# and lm.score(X_std_train, y_std_train)>0.25 and MAE(y_true=y_test, y_pred=y_predict)<0.25):
        print('Mean Absolute Error: ', MAE(y_true=y_test,y_pred=y_predict))
        print('R2 of training data: ', lm.score(X_std_train, y_std_train))
        print('% Gross Errors: ', countGrossErrors(y_test,y_predict)/len(y_predict))
        print('Random seed for tts: ', randState)
        plot_parity(x=y_test, y=y_predict, labels=None, xlabel='True Selectivity',ylabel='Predicted Selectivity')

    return lm.score(X_std_train, y_std_train),MAE(y_true=y_test, y_pred=y_predict),countGrossErrors(y_test,y_predict)/len(y_predict)

def doKMeansClusteringWithRBFKPCAWith3Components(X,y,sample_weights,vectrs=[0,1,2]):
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
    ''' this plot shows that 4 clusters is optimal
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
    '''
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

if __name__ == '__main__':
    X, y, feature_weights, sample_weights, ligNames =  loadData(zeroReplace=0.01)
    # familySeparation(dc(X),dc(y),dc(ligNames))
    # R2s=[]; maes=[]; GEs=[];
    # for i in range(0,1000):
        # R2, mae, GE = doWPCA(dc(X),dc(y),dc(feature_weights),dc(sample_weights),output=False)
        # R2, mae, GE = doRidgeCV(dc(X),dc(y),dc(feature_weights),dc(sample_weights),output=True)
        # R2, mae, GE = doLASSO(dc(X),dc(y),dc(feature_weights),dc(sample_weights),output=True)
        # R2, mae, GE = doKPCA(dc(X),dc(y),output=True)
        # R2s.append(R2); maes.append(mae); GEs.append(GE);
    # doKPCASeparation(dc(X),dc(y))
    # doKMeansClusteringWithRBFKPCAWith3Components(dc(X),dc(y),dc(sample_weights))
    # doKMeansClustering(dc(X),dc(y),dc(sample_weights))
    
    # plt.figure()
    # plt.hist(GEs,rwidth=0.9,bins=[i/20 for i in range(0,20)],align='left')
    # plt.title('RBF-KPCA Distribution of %GE (1000 runs, 80:20 training:test)')
    # plt.ylabel('Frequency')
    # plt.xlabel('%GE')
    # plt.xticks([i/20 for i in range(0,20)],[str(i) for i in range(0,96,5)])
    # plt.show()
    # print("""Average %GE: {:.2f} Standard Deviation: {:.2f}""".format(np.average(GEs),np.std(GEs)))

"""
3-Means Clustering on Scaled Data:
Cluster 0 average and stdev:
0.30469411214281666
0.40053104923738647
Max in this Cluster:  1.0
Cluster 1 average and stdev:
-0.23992603326009596
0.4550882101556402
Max in this Cluster:  0.9242463079717002
Cluster 2 average and stdev:
-0.38072314747564323
0.3468476014150291
Max in this Cluster:  0.1530147587934812

K-Means Clustering on RBF 4-Component PCA
Cluster 0 (red) average and stdev:
0.36942100042981535
0.35070508549836693
Cluster 1 (green) average and stdev:
-0.23043421338312967
0.5191292407314995
Cluster 2 (blue) average and stdev:
-0.2162061107937193
0.4289383248262524
Cluster 3 (orange) average and stdev:
-0.3453850894798298
0.3604713124172736


RBF KPCA w/ 3 Components:
Mean Absolute Error:  0.25915683583389704
R2 of training data:  0.3532508522886406
% Gross Errors:  0.0
Random seed for tts:  270252416
Mean Absolute Error:  0.26303986364507775
R2 of training data:  0.3689261497471472
% Gross Errors:  0.0
Random seed for tts:  220232570
Mean Absolute Error:  0.2843560462696692
R2 of training data:  0.3455074211212217
% Gross Errors:  0.0
Random seed for tts:  945107195
Mean Absolute Error:  0.3194705614025062
R2 of training data:  0.3391373329330386
% Gross Errors:  0.0
Random seed for tts:  536367301
Average %GE: 0.26 Standard Deviation: 0.09

RBF KPCA w/ 4 Components:
Mean Absolute Error:  0.2819107574618129
R2 of training data:  0.3977701391835511
% Gross Errors:  0.0
Random seed for tts:  449119043
Mean Absolute Error:  0.30894735554882496
R2 of training data:  0.35324491555535087
% Gross Errors:  0.0
Random seed for tts:  82702285
Mean Absolute Error:  0.2205499763350409
R2 of training data:  0.37968097604140394
% Gross Errors:  0.0
Random seed for tts:  776208407
Average %GE: 0.26 Standard Deviation: 0.09

WPCR:
Best results so far:
0.9505432719216261 explained by 3 eigen-vectors
Mean Absolute Error:  0.2533452596352515
R2 of training data:  0.2707783943385329
% Gross Errors:  0.0
Random seed for tts:  632130344

0.9662726637817706 explained by 5 eigen-vectors
Mean Absolute Error:  0.23734940231762897
R2 of training data:  0.260765038246244
% Gross Errors:  0.0
Random seed for tts:  766178200

0.9587033761934431 explained by 4 eigen-vectors
Mean Absolute Error:  0.23388469610427456
R2 of training data:  0.27047264455249775
% Gross Errors:  0.0
Random seed for tts:  355646495

Distribution of MAE's:
Average %GE: 0.25 Standard Deviation: 0.09

RidgeCV:
0.37+-.10

LASSO:
Average %GE: 0.25 Standard Deviation: 0.09

RBF KPCA:
Average %GE: 0.36 Standard Deviation: 0.10

Cosine KPCA:
Average %GE: 0.37 Standard Deviation: 0.11

Poly d2:
Average %GE: 0.37 Standard Deviation: 0.10
Poly d3:

"""