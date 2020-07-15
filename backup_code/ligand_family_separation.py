"""
@uthor: Himaghna, 30th September 2019
Description: Look at separation of families in space of first three eigenvectors
"""

from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
import pickle

import data_processing


# processed_data = data_processing.main(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\data_proc_config.json')
# X, y = processed_data['X'], processed_data['y']
# descriptor_names = processed_data['descriptor_names']
# family_column = processed_data['family_int']

def do_family_separation(vectrs=[0, 1, 2]):
    """
    Look at separation obtained between ligand families by the top n_eigenvectors
    of the data matrix
    Params ::
    vectrs: List(int), size 3: Indexes of eigenvectors to retain. 
       Default [0, 1 ,2]
    """

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)


    pca_reduced = PCA()
    X_pca = pca_reduced.fit_transform(X_std)
    
    # plot 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'red', 'orange', '#FF00F0','black']
    ligand_names = ['4-L28','3-L12','4-L23','4-L11','4-L15','4-L16','4-L24','1-L13','3-L8','4-L1','4-L19','4-L12','2-L13','4-L20','4-L21','3-L4','3-L10','3-L9','4-L9','4-L17','3-L3','4-L10']
    family_column = [0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2]
    #plt.legend()
    c = [colors[int(family)] for family in family_column] # conditional coloring
    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.5, s=20)
    # print("xdata")
    # print(X_pca[:, vectrs[0]])
    # print("ydata")
    # print(X_pca[:, vectrs[1]])
    # print("zdata")
    # print(X_pca[:, vectrs[2]])
    # write ligand labels
    for (x, y, z) in zip(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]]):
        ax.text(
            x,
            y,
            z,
            ligand_names.pop(0),
            zdir=(1,1,1),
            fontsize=10
        )

    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    plt.show()
    print('Max PC1', y[np.argmax(X_pca[:, 0])])

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
    do_family_separation()
    