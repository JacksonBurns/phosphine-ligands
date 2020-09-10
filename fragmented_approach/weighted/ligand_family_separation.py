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
    ligand_names = ['1.L1','1.L11','1.L12','1.L13','1.L2','1.L3','1.L4','1.L5','1.L6','1.L7','1.L8','1.L9','2.L1','2.L10','2.L11','2.L12','2.L13','2.L14','2.L15','2.L16','2.L17','2.L18','2.L19','2.L2','2.L21','2.L22','2.L24','2.L25','2.L26','2.L27','2.L28','2.L29','2.L3','2.L4','2.L7','2.L9','3.L1','3.L10','3.L11','3.L12','3.L14','3.L15','3.L16','3.L2','3.L3','3.L4','3.L5','3.L6','3.L7','3.L8','3.L9','4.L1','4.L10','4.L11','4.L12','4.L13','4.L14','4.L15','4.L16','4.L17','4.L18','4.L19','4.L2','4.L20','4.L21','4.L22','4.L23','4.L24','4.L25','4.L26','4.L27','4.L29','4.L3','4.L4','4.L5','4.L6','4.L8','4.L9']
    family_column = [2,3,3,0,2,2,1,1,2,3,2,2,1,2,2,2,3,2,3,2,2,3,2,3,2,2,3,2,2,3,2,2,2,2,2,2,2,0,0,0,2,0,2,3,1,0,0,3,1,0,2,0,1,0,1,2,2,0,0,0,0,0,2,2,0,1,1,0,2,2,0,1,2,2,2,0,0,1]
    #plt.legend()
    c = [colors[int(family)] for family in family_column] # conditional coloring
    ax.scatter(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]], \
        c=c, alpha=0.5, s=100)
    # print("xdata")
    # print(X_pca[:, vectrs[0]])
    # print("ydata")
    # print(X_pca[:, vectrs[1]])
    # print("zdata")
    # print(X_pca[:, vectrs[2]])
    # write ligand labels
    for (x, y, z) in zip(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]]):
        ax.text(x,y,z,
            ligand_names.pop(0),
            zdir=(1,1,1),
            fontsize=10)

    ax.set_xlabel(f'PC {vectrs[0] + 1}', fontsize=20)
    ax.set_ylabel(f'PC {vectrs[1] + 1}', fontsize=20)
    ax.set_zlabel(f'PC {vectrs[2] + 1}', fontsize=20)
    plt.show()
    print('Max PC1', y[np.argmax(X_pca[:, 0])])

if __name__ == '__main__':
    X = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\X.p', "rb"))
    y = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\y.p', "rb"))
    feature_weights = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\feature_weights.p', "rb"))
    sample_weights = pickle.load(open(r'C:\Users\jwb1j\OneDrive\Documents\GitHub\phosphine-ligands\fragmented_approach\weighted\sample_weights.p', "rb"))
    do_family_separation()
    