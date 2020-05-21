"""
@uthor: Himaghna, 22nd October 2019
Description: Get similarity
"""

import os.path

from argparse import ArgumentParser
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from seaborn import heatmap, kdeplot
 

class Molecule:
    """
    molecular object defined from rdkit mol object
    """
    def __init__(self, mol, name_=None):
        """
        mol: RDkit mol object
        """
        self.mol = mol
        self.name_ = name_
    
    def get_morgan_fingerprint(self, radius=3, n_bits=None):
        """
        generate a morgan fingerprint
        Params ::
        radius: int: radius of fingerprint, 3 corresponds to diameter 6.
                    Default 3
        n_bits: int: Number of bits to use if Morgan Fingerprint wanted as a bit
            vector. If set to None, Morgan fingerprint returned as count.
            Default is None
        Returns ::
        morgan_fingerprint
        """
        if n_bits is None:
            return AllChem.GetMorganFingerprint(self.mol, radius)
        else:
            return AllChem.GetMorganFingerprintAsBitVect(self.mol, radius, \
                nBits=n_bits)
    
    def get_similarity_to_molecule(self, target_mol, similarity_measure='tanimoto'):
        """
        get a similarity metric to a target molecule
        Params ::
        target_mol: Molecule object: Target molecule. 
            Similarity score is with respect to this molecule
        similarity_measure: str: the similarity metric used.
            *** Supported Metrics ***
            'tanimoto': Jaccard Coefficient/ Tanimoto Similarity 
                    0 (not similar at all) to 1 (identical)
        
        Returns ::
        similarity_score: float: similarity coefficient by the chosen method
        """


        if similarity_measure == 'tanimoto':
            similarity_measure = DataStructs.TanimotoSimilarity(\
                self.get_morgan_fingerprint(), \
                    target_mol.get_morgan_fingerprint())
        
        return similarity_measure
    

def generate_distance_matrix(mols, **kwargs):
    """
    Get a pairwise similarity matrix
    Params ::
    mols: List[Molecules]: collection of input molecules
    **kwargs: additional aguments to supply. Important ones are
        ***similarity_measure*** used by get_similarity_to_molecule
    Returns ::
    distance_matrix: np ndarray: n_mols X n_mols numpy matrix containing 
        pairwise similarity scores
    """
    n_mols = len(mols)
    distance_matrix = np.zeros(shape=(n_mols, n_mols))
    for id, mol in enumerate(mols):
        for target_id in range(id, n_mols):
            distance_matrix[id, target_id] = mol.get_similarity_to_molecule(mols[target_id])
            # symmetric matrix entry
            distance_matrix[target_id, id] = distance_matrix[id, target_id]
    return distance_matrix

def draw_distance_heatmap(distance_matrix, **kwargs):
    """
    Plot a heatmap of the distance matrix
    Params ::
    distance_matrix: np ndarray: n_mols X n_mols numpy matrix containing 
        pairwise similarity scores
    
    Returns ::
    None
    """
    xticklabels = kwargs.get('xticklabels', False)
    yticklabels = kwargs.get('yticklabels', False)
    cmap = kwargs.get('cmap', 'autumn')
    mask_upper = kwargs.get('mask_upper', True)
    annotate = kwargs.get('annotate', True)
    plt.rcParams['svg.fonttype'] = 'none'
    mask = None
    if mask_upper is True:
        mask = np.triu(np.ones(shape=(distance_matrix.shape)), k=0)
    heatmap_obj = heatmap(distance_matrix, xticklabels=xticklabels, \
        yticklabels=yticklabels, cmap=cmap, mask=mask, annot=annotate)
    plt.show()

def plot_density(vector_, **kwargs):
    """
    Plot density of some vector such as similarity of molecule with dataset
    Params ::
    vector_: n x 1 array like object: desnity input
    **kwargs: dict: keywords modifying behavior of the plot
    Returns ::
    None
    """
    plt.rcParams['svg.fonttype'] = 'none'
    kdeplot(vector_, **kwargs)
    plt.show()

def draw_similarity_distr(distance_matrix, **kwargs):
    """
    Plot the distribution of the similarities as present in the lower diagonal
    values (diagonal not included) of the distance matrix
    Params ::
    distance_matrix: np ndarray: n_mols X n_mols numpy matrix containing 
        pairwise similarity scores
    **kwargs: dict: keyword arguments. Passed also to thefunction plot_density()
    
    Returns ::
    None
    """
    lower_diag_indices = np.tril_indices(distance_matrix.shape[0], -1)
    similarity_vector = distance_matrix[lower_diag_indices]
    plot_density(similarity_vector, **kwargs)





def main():
    """
    Take in directory of .mol files or text of SMILES and generate Similarity
    Maps
    """
    parser = ArgumentParser()
    parser.add_argument('mol_location', \
        help='Put in either a text file containing list of SMILES or a \
            folder containing .pdb/.mol files')
    parser.add_argument('-tf', '--target_file', required=False, default=None, \
        help='Filepath of target molecule. If set, similarity scores of ' \
            'target molecule with all molecules in the <<mol_location>> '\
                'folder  is calculated and displayed')
    args = parser.parse_args()
    mol_location = args.mol_location
    target_file = args.target_file
    if os.path.isdir(mol_location):
        print(f'Searching for *.pdb files in {mol_location}')
        mol_list = []
        for molfile in glob(os.path.join(mol_location, '*.pdb')):
            mol_object = Chem.MolFromPDBFile(molfile)
            mol_name = os.path.basename(molfile).replace('.pdb', '')
            if mol_object is None:
                print(f'{molfile} could not be imported. Skipping')
                continue
            mol_list.append(Molecule(mol_object, mol_name))
    else:
        raise FileNotFoundError(f'{mol_location} could not be found.' \
            'Please enter valid foldername or path of a text file' \
                'containing SMILES strings')
    if len(mol_list) == 0:
        print('No molecular files found in the location!')
        exit()
    if target_file is None:
        # heatmap of data required
        distance_matrix = generate_distance_matrix(mol_list)
        draw_distance_heatmap(distance_matrix, cmap='YlGnBu_r', \
            mask_upper=False, annotate=True)
        # plot distribution of similarities
        draw_similarity_distr(distance_matrix, shade=True, color='red', bw=0.01)
    else:
        # vector of similarity of target molecule to all other molecules needed
        target_mol_object = Chem.MolFromPDBFile(target_file)
        if target_mol_object is None:
            print(f'Target {molfile} could not be imported. Exiting')
            exit()
        mol_name = os.path.basename(target_file).split('.')[0]
        target_molecule = Molecule(target_mol_object, mol_name)
        target_similarity = [target_molecule.get_similarity_to_molecule(ref_mol) \
            for ref_mol in mol_list]
        print(f'*****FOR MOLECULE {target_molecule.name_}*****')
        print('****Maximum Similarity Molecule ****')
        print(mol_list[np.argmax(target_similarity)].name_, \
            target_similarity[np.argmax(target_similarity)])
        print('****Minimum Similarity Molecule ****')
        print(mol_list[np.argmin(target_similarity)].name_, \
            target_similarity[np.argmin(target_similarity)])
        plot_density(target_similarity, shade=True, color='violet', bw=0.01)

    


if __name__ == '__main__':
    main()
    
    
    





    





    

        
        
        

    

