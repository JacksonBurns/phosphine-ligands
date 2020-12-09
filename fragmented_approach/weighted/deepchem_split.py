import deepchem as dc
import numpy as np



with open(r'ml_ligands_as_smiles.txt','r') as file:
    data = np.array(file.readlines())
# loop through, replace each with itself.replace("\n","")

splitter = dc.splits.ButinaSplitter(0.6)
ret = splitter.train_test_split(dataset=data)
print(ret)