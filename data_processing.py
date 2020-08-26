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


def process_data(df, target_column, descrptr_columns):
    """
    Separate DataFrame into the y (target) vector and X (descriptors) matrix
    Ensure only rows with non-nan values for targets and some columns are kept
    Params:
    df DataFrame object
    target_column (str)
    descrptr_columns array_like_object(str): list of columns as descriptors
    """
    print(f'Original shape of DF: {df.shape}')

    # drop target NaN rows
    df.dropna(subset=[target_column], inplace=True)
    print(f'Shape of DF after dropping empty targets: {df.shape}')

    # drop rows where all descriptor values except 'Family' are NaN
    df.dropna(subset=[_ for _ in descrptr_columns if not _ == 'Family'],
               inplace=True, how='all')
    print(f'Shape of DF after dropping empty targets: {df.shape}')
    return df


def generate_input(df, descrptr_columns, bins_upper_limit):
    """
    Make the input matrix by suitably treating the various descriptor columns
    Params ::
    df: Pandas DataFrame: Original Pandas datafram containing all the data
    descrpt_columns: List: All the columns treated as descriptors
    bins_upper_limits: List: Upper limits (non-inclusive) of the bins 
    (sorted from min to max)
    e.g. for bin_upper_limits = [x, y], 
    bins are j | [<x, x<=j <y, y <= j]
    Return ::
    X: numpy ndarray: Suitably transformed input  
    descriptor_names: List[str]: Descriptor names for columns of X
    """

    # Define Globals
    frequency_void_flag = -1
    # frequency_columns = [ f'IR Frequency{frequency_index}' \
    #     for frequency_index in range(1, 7)]
    bond_length_columns = [f'Bond Lengths_angstrom{index}' \
        for index in range(1,4)]
    bond_angle_columns = [f'Bond Angles_degrees{index}' \
        for index in range(1,4)]
    mulliken_charge_columns = [f'Mulliken_Charges{index}' \
        for index in range(1,6)]
    sterimol_parameter_columns = [f'Sterimol_Parameters{index}' \
        for index in range(1,4)]
    apt_charge_columns = [f'APT Charge{index}' \
        for index in range(1,6)]

    # transform NaN frequencies to frequency_void_flag
    # for column in frequency_columns:
    #     df.loc[df[column].isna(), column] = frequency_void_flag
    
    def analyze_frequencies(show_plot=False):
        """
        Visualize frequencies to help in binning
        """
        frequencies = []
        for column in frequency_columns:
            freqs = df[column].values
            if frequencies is None:
                frequencies.append(freqs)
            else:
                frequencies.extend(freqs)
        # take out missing frequencies containing void_flag values
        frequencies = [_ for _ in  frequencies if not _ == frequency_void_flag]
        if show_plot is not False:
            plt.scatter([_ for _ in range(len(frequencies))], frequencies)
            plt.show()
            plt.hist(frequencies)
            plt.xlabel('Frequencies (cm -1)')
            plt.ylabel('Count')
            plt.show()

    def bin_frequencies(frequencies):
        """
        Create a new descriptor based on number of occurences of frequency in 
        bins defined by the upper bounds (non inclusive)
        Params ::
        frequencies: s x f numpy array : Array containing the f frequencies of 
            s s

        void_flag: int: signals missing frequency. Default=-1
        
        Returns ::
        binned_frequencies: s x total_bins numpy array: transformed descriptors
            containing the binned distribution of frequencies for each sample
        """
        s, _ = frequencies.shape
        total_bins = len(bins_upper_limit) + 1
        binned_frequencies = []
        def get_bin_idx(freq):
            """
            Get the id of the bin that freq falls into
            Params ::
            freq: float: freq to bin
            Returns ::
            bin_id: float: id of the bin, zero indexed. 
            Possible values 0 -> len(bin_upper_limits)
            """
            for bin_id, upper_limit in enumerate(bins_upper_limit):
                if freq < upper_limit:
                    return bin_id
            # for frequencies >= greatest bin value
            bin_id = total_bins - 1
            return bin_id

        for sample_num in range(s):
            row = frequencies[sample_num, :]  # array size f
            sample_binned_frequencies = [0] * total_bins
            for freq in row:
                if freq == frequency_void_flag:
                    continue
                sample_binned_frequencies[get_bin_idx(freq)] += 1
            binned_frequencies.append(sample_binned_frequencies)
        
        binned_frequencies = np.array(binned_frequencies).reshape(s, total_bins)
        return binned_frequencies
    
    # generate numpy array of descriptors
    frequencies = df[frequency_columns].to_numpy()
    # s x total_bins
    binned_frequencies = bin_frequencies(frequencies) 

    # average bond lengths, angles, mulliken and apta charges, sterimol
    bond_length = np.mean(df[bond_length_columns].to_numpy(), \
        axis=1).reshape(-1, 1)
    bond_angle = np.mean(df[bond_angle_columns].to_numpy(), \
        axis=1).reshape(-1, 1)
    mulliken_charge = np.mean(df[mulliken_charge_columns].to_numpy(), \
        axis=1).reshape(-1, 1)
    sterimol_parameter = np.mean(df[sterimol_parameter_columns].to_numpy(), \
        axis=1).reshape(-1, 1)
    apt_charge = np.mean(df[apt_charge_columns].to_numpy(), \
        axis=1).reshape(-1, 1)

    other_descriptor_columns = np.setdiff1d(descrptr_columns, \
        frequency_columns + bond_length_columns + bond_angle_columns \
        + mulliken_charge_columns + sterimol_parameter_columns \
        + apt_charge_columns)
    
    other_descriptors = df[other_descriptor_columns].to_numpy()

    X = np.concatenate((binned_frequencies, bond_length, bond_angle, \
        mulliken_charge, sterimol_parameter, apt_charge, other_descriptors), \
            axis=1)
    descriptor_names = ['binned_frequency'] * binned_frequencies.shape[1] + \
        ['average_bond_length', 'average_bond_angle', \
            'average_mulliken_charge', 'average_sterimol_parameter', \
                'average_apt_charge'] + list(other_descriptor_columns)
    return X, descriptor_names


def impute_missing_values(X, method='median'):
    """
    Impute the missing values in data matrix (nan) with a fill-in value which
    is either the mean or median of the observations in that column
    Params ::
    X: s x d numpy nadarray: Data matrix of s samples and d descriptors
    method: str: Method used to impute the missing values. Options are 'median'
        'mean'. Default is 'median'
    
    Returns ::
    X: s x d numpy ndarray: Data matrix with nan values imputed
    """
    if method not in ['mean', 'median']:
        raise NotImplementedError(f'{method} not implementesd')
    impute_value = np.nanmean(X, axis=0) if method == 'mean' else \
        np.nanmedian(X, axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(impute_value, inds[1])
    return X


def onehot(column):
    """
    Convert ordinal column to one hot encoded descriptors
    Params ::
    column: s x 1 np array: array of ordinal observations with c unique values
    Returns ::
    onehot_encoded: s X c np array: output array of observations, each of size c
    """
    categories = list(set(column))
    n_categories = len(categories)
    list_one_hot = []
    for observation in column:
        sample = [0] * n_categories
        sample[categories.index(observation)] = 1
        list_one_hot.append(sample)
    return np.array(list_one_hot).reshape(len(column), -1)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_proc_config')
    args = parser.parse_args()

    configs = json.load(open(args.data_proc_config))
    xl_file = configs.get('xl_file')
    target_column = configs.get('target_column')
    descrptr_columns = configs.get('descrptr_columns')
    bins_upper_limit = configs.get('bins_upper_limit')
    out_dir = configs.get('output_directory')

    df = pd.read_excel(xl_file)
    df = process_data(df, target_column, descrptr_columns)
    y = df[target_column].values
    # X, descriptor_names = generate_input(df, descrptr_columns, bins_upper_limit)
    # X = impute_missing_values(X, method='median')
    descriptor_names = descrptr_columns
    X = df[descrptr_columns].to_numpy()
    out_dict = {
        'X': X,
        'y': y,
        'descriptor_names': descriptor_names
    }

    print('******* STATISTICS ********')
    print(f'*** {target_column} ***')
    print(f'Mean: {np.mean(y)}')
    print(f'Std. Dev.: {np.std(y)}')
    plt.rcParams['svg.fonttype'] = 'none'
    plt.hist(y, color='orange')
    plt.xlabel(target_column, fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.show()
    
    X_path = os.path.join(out_dir, 'X.p')
    print(f'Writing {X_path}...')
    pickle.dump(X, open(X_path, "wb"))
    y_path = os.path.join(out_dir, 'y.p')
    print(f'Writing {y_path}...')
    pickle.dump(y, open(y_path, "wb"))
    descriptor_names_path = os.path.join(out_dir, 'descriptor_names.p')
    print(f'Writing {descriptor_names_path}...')
    pickle.dump(descriptor_names, open(descriptor_names_path, "wb"))



