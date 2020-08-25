"""
author: Nils Beck, nils.beck@pm.me

Evidence Project
@ Ubiquitous Knowledge Processing Lab
@ TU Darmstadt, Germany

May 2020

In this file I train a multilayer perceptron (MLP) - a simple version of a neural network.
The model will read sentences (or rather sentence embeddings) and compute a score, representing how well they are suited
as example sentences for a dictionary
"""
import torch
from experiments import mlp_lib
import numpy as np
from datetime import datetime

EMBEDDINGS_PATH = '/ukp-storage-1/nbeck/data/evidence/embeddings.npy'
LABELS_PATH = '/ukp-storage-1/nbeck/data/evidence/labels.npy'

# load and partition the data sets
train_set, dev_set, test_set = mlp_lib.partition_data(np.load(EMBEDDINGS_PATH), np.load(LABELS_PATH), 0.9, 0.05)

# create a MLP
mlp = mlp_lib.MLP()

# train the MLP
mlp_lib.train(mlp, train_set, dev_set, 50)

# save the trained model
model_name = '/ukp-storage-1/nbeck/models/mlp_' + str(datetime.now())
torch.save(mlp.state_dict(), model_name)


# test the trained MLP
mlp_lib.test(mlp, test_set)



