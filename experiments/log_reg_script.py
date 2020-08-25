"""
author: Nils Beck, nils.beck@pm.me

Evidence Project
@ Ubiquitous Knowledge Processing Lab
@ TU Darmstadt, Germany

May 2020

In this file I train a logistic regression.
This model reads sentence embeddings and maps them to a label in {0, 1, 2, 3},
representing how well-suited each sentence is for being used as a dictionary example sentence
"""
import collections

from experiments import log_reg_lib
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from datetime import datetime

EMBEDDINGS_PATH = '/ukp-storage-1/nbeck/data/evidence/embeddings.npy'
LABELS_PATH = '/ukp-storage-1/nbeck/data/evidence/labels.npy'
MODEL_PATH = '/ukp-storage-1/nbeck/models/'

# create a logistic regression model
model = LogisticRegression(max_iter=10000)

# retrieve our data sets
all_embeddings = np.load(EMBEDDINGS_PATH)
all_labels = np.load(LABELS_PATH)
train_set, test_set = log_reg_lib.partition_data(all_embeddings, all_labels)

# train the model
X_train, y_train = train_set
model.fit(X_train, y_train)

# test and evaluate the model
log_reg_lib.test(model, test_set)

# print class distribution for comparison
print('Occurrence of classes in whole set: \n' + str(collections.Counter(all_labels)))
print('Occurrence of classes in training set: \n' + str(collections.Counter(y_train)))
print('Occurrence of classes in test set: \n' + str(collections.Counter(test_set[1])))

# store the model in a file
joblib.dump(model, MODEL_PATH + 'log_reg_' + str(datetime.now()) + '.pkl')
