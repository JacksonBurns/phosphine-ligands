"""@uthor: Himaghna, 4th September, 2019
Description: Process data

"""
from argparse import ArgumentParser
import os.path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_proc_config')
    args = parser.parse_args()

    configs = json.load(open(args.data_proc_config))
    xl_file = configs.get('xl_file')
    target_column = configs.get('target_column')
    descrptr_columns = configs.get('descrptr_columns')
    bins_upper_limit = configs.get('bins_upper_limit')
    err_column = configs.get('error_column')
    out_dir = configs.get('output_directory')

    df = pd.read_excel(xl_file)
    y = df[target_column].values
    descriptor_names = descrptr_columns
    X = df[descrptr_columns].to_numpy()

    # pull and process weight column into shape of input array
    weights = df[err_column].to_numpy()  # read it
    weights = 1./weights  # invert
    weights = weights[np.newaxis]  # change from 1D to 2D
    weights = weights.T  # transpose to column vector
    sample_weights = weights
    feature_weights = np.repeat(weights, len(df[descrptr_columns].columns), axis=1)  # copy columns across to match input data

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



