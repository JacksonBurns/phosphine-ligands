import numpy
import pandas

from chainer_chemistry.dataset.parsers.data_frame_parser import DataFrameParser  # NOQA
from chainer_chemistry.dataset.preprocessors import AtomicNumberPreprocessor
from chainer_chemistry.dataset.splitters.deepchem_scaffold_splitter import generate_scaffold  # NOQA
from chainer_chemistry.dataset.splitters.deepchem_scaffold_splitter import DeepChemScaffoldSplitter  # NOQA
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

def get_scaffold_idxs():
    with open(r'ligands_as_smiles.smi','r') as file:
        smiles = file.readlines()
        smiles_list = [smi.replace("\n","") for smi in smiles]


    df = pandas.DataFrame(data={'smiles': smiles_list,
                                'value': numpy.random.rand(len(smiles_list))})
    pp = AtomicNumberPreprocessor()
    parser = DataFrameParser(pp, labels='value')
    dataset = parser.parse(df, return_smiles=True)

    splitter = DeepChemScaffoldSplitter()
    train, valid = splitter.train_valid_split(dataset=dataset['dataset'],
                                                smiles_list=dataset['smiles'],
                                                return_index=True,
                                                frac_train=0.8, frac_valid=0.2)
    return list(train), list(valid)

# train, test = get_scaffold_idxs()

# with open(r'ligands_names.txt','r') as file:
#     names = file.readlines()
#     names = [i.replace("\n","") for i in names]

# print("Scaffold-based Training: ",[names[i] for i in train])
# print("Scaffold-based Testing: ",[names[i] for i in test])