"""
@uthor: Himaghna, 19th September 2019
Description: Perform exploratory data analysis on the data-set
"""

from argparse import ArgumentParser
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# pyplot parameters
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

def plot_pairplot(skip_frequency=True, save_location=None):
    """
    Plot the satter plot for each pair of descriptors
    Params ::
    skip_frequency: Boolean: True if frequency variables not considered.
        False otherwise
    save_location: Boolean: Folder to save plots. If set to None, 
        then plot is displayed. Default value is None
    Returns ::
    None
    """
    for index1 in range(X.shape[1]-1):
        if skip_frequency and descriptor_names[index1] == 'binned_frequency':
            continue
        for index2 in range(index1+1, X.shape[1]):
            if skip_frequency and descriptor_names[index1] == 'binned_frequency':
                continue
            plt.scatter(X[:, index1], X[:,index2], alpha=0.7)
            plt.xlabel(descriptor_names[index1], fontsize=20)
            plt.ylabel(descriptor_names[index2], fontsize=20)
            if save_location is None:
                plt.show()
            else:
                fname = f'{descriptor_names[index2]}_vs_{descriptor_names[index1]}.svg'
                print(f'Saving {fname}')
                plt.savefig(os.path.join(save_location, fname))
            plt.close()


def do_PCA(save_location=None, variance_needed=0.90):
    """
    Do PCA on the data matrix X
    Params ::
    save_location: Boolean: Folder to save plots. If set to None, 
        then plot is displayed. Default value is None
    Returns ::
    n_eigenvectors_needed: int: Number of eigenvectors to explain variance_needed
    """
    scaler = StandardScaler()
    pca = PCA()



    X_std = scaler.fit_transform(X)
    
    pca.fit(X_std)
    
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
    plt.plot([i for i in range(1, len(cum_var)+1)], cum_var, color='#E5FF3F',
            linewidth=2.0)
    plt.xlabel('Eigenvector', fontsize=20)
    plt.ylabel('Ratio of variance explained', fontsize=20)
    plt.title('Principal Component Analysis', fontsize=24)
    for key, value in enumerate(cum_var):
        if value >= variance_needed:
            n_eigenvectors_needed = (key+1)
            print('{} explained by {} eigen-vectors'.format(value,
                                                            n_eigenvectors_needed))
            break
    plt.hlines(y=1.0, xmin=0, xmax=len(cum_var), color='black', linestyles='--',
            alpha=0.5)
    plt.plot([i for i in range(1, n_eigenvectors_needed+1)],
            cum_var[:n_eigenvectors_needed], color='#FF23A5', linewidth=2.0)
    assert cum_var[n_eigenvectors_needed-1] == value
    plt.text(x=n_eigenvectors_needed, y=cum_var[n_eigenvectors_needed-1]-0.1,
            s='{} Eigenvectors explain {} of variance'.
            format(n_eigenvectors_needed,
                    round(cum_var[n_eigenvectors_needed-1], 2)),
            color='#C7157E', fontsize=20)
    if save_location is None:
        plt.show()
    else:
        fname = f'PCA_{variance_needed}.svg'
        print(f'Saving {fname}')
        plt.savefig(os.path.join(save_location, fname))
    plt.close()
    return n_eigenvectors_needed



def check_correlation_with_target(descriptor_name):
    """
    Plot how a descriptor varies with the target
    Params ::
    descriptor_name: str: column name corresponding to the descriptor
    Returns ::
    None
    """
    plt.scatter(X[:, descriptor_names.index(descriptor_name)], y, alpha=0.7)
    plt.xlabel(descriptor_name, fontsize=20)
    plt.ylabel('Target', fontsize=20)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-x', help='Path of X.p')
    parser.add_argument('-y', help='Path of y.p')
    parser.add_argument('-dn', '--descriptor_names', 
                        help='Path of descriptor_names.p')
    args = parser.parse_args()
    
    X = pickle.load(open(args.x, "rb"))
    y = pickle.load(open(args.y, "rb"))
    descriptor_names = pickle.load(open(args.descriptor_names, "rb"))
    
    # for col in range(X.shape[1]):
    #     # plt.hist(X[:, col], color='green')
    #     # plt.ylabel('Frequency', fontsize=28)
    #     # plt.xlabel(descriptor_names[col], fontsize=28)
    #     # plt.xticks(fontsize=20)
    #     # plt.yticks(fontsize=20)
    #     # plt.show()
    #     # continue
    #     plt.scatter(X[:, col], y, c='red', alpha=0.4, s=100)
    #     plt.xlabel(descriptor_names[col], fontsize=20)
    #     plt.ylabel('Response', fontsize=20)
    #     plt.show()

    
    # plot_pairplot(save_location=r'C:\Users\jwb1j\OneDrive\Desktop\Working Folder\Research 2\ML Collab\Week 4\Figures\traditional')
    n_eigenvectors = do_PCA(save_location=r'C:\Users\jwb1j\OneDrive\Desktop\Working Folder\Research 2\ML Collab\Week 4\Figures\traditional', \
       variance_needed=0.95)









