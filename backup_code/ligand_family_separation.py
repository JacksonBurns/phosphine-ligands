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
    ligand_names = ['1-1','1-2','1-4','1-6','1-7','1-8','1-9','1-10','1-11','1-12','1-13','2-1','2-2','2-3','2-4','2-5','2-7','2-8','2-9','2-10','2-12','2-13','2-14','2-15','2-16','2-20','2-21','2-22','2-23','2-24','2-25','2-26','2-27','2-29','2-30','2-31','3-1','3-2','3-3','3-4','3-5','3-6','3-8','3-9','3-10','3-12','3-13','3-16','4-1','4-2','4-3','4-4','4-5','4-6','4-7','4-8','4-9','4-10','4-11','4-12','4-13','4-14','4-15','4-16','4-17','4-18','4-19','4-20','4-21','4-24','4-25','4-26','4-27','4-28','4-29','4-30','4-31']
    family_column = [2,2,2,2,2,2,2,2,2,3,2,2,2,2,2,2,2,3,2,2,2,2,2,2,2,2,2,2,2,3,2,2,0,2,2,2,2,2,1,1,2,2,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,1,2,2,2,2,2,2,2,3,2,1,2,2,0]
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
    # for (x, y, z) in zip(X_pca[:, vectrs[0]], X_pca[:, vectrs[1]], X_pca[:, vectrs[2]]):
    #     ax.text(x,y,z,
    #         ligand_names.pop(0),
    #         zdir=(1,1,1),
    #         fontsize=10)

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
    