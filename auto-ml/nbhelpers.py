import types, json
import pandas as pd
import numpy as np
from dcekit.sampling import kennard_stone 

def loadData(config, zeroReplace=-1, removeLessThan=None, removeGreaterThan=None):
    """
    Retrieves data from Excel file and optionally writes it out to serialized format
    """
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

    return X, y, feature_weights, sample_weights, ligNames

def splitData(ttd, randState=None, splitter=None, trainSize=0.80):
    # split response, features, and weights into train and test set
    if randState is None:
        randState = random.randint(1,1e9)
        print('Random Seed: ',randState)
    if splitter is None:
        X_train, X_test, y_train, y_test, f_weights_train, f_weights_test, s_weights_train, s_weights_test, ln_train, ln = train_test_split(ttd.X, ttd.y, ttd.f_weights, ttd.s_weights, ttd.ln, train_size=trainSize, random_state=randState)
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