"""
author: Nils Beck, nils.beck@pm.me

Evidence Project
@ Ubiquitous Knowledge Processing Lab
@ TU Darmstadt, Germany

May 2020

In this file I provide all necessary functionality for my logistic regression script
This model reads sentence embeddings and maps them to a label in {0, 1, 2, 3},
representing how well-suited each sentence is for being used as a dictionary example sentence
"""
import random
from sklearn.metrics import f1_score, confusion_matrix


def partition_data(X, y):
    """
    partition the given lists, containing embeddings and labels respectively, into
    - train set (80%)
    - test set  (20%)
    also we are going to make sure the test set contains 20% of each class' occurrences
    :param X: embeddings list
    :param y: labels list
    :return: two array tuples, containing the train and test data in the form
    (embeddings , labels)
    """
    # represent data as a dictionary with fields for each class
    class_dict = {0: [], 1: [], 2: [], 3: []}
    for i in range(len(X)):
        class_dict[int(y[i])].append(X[i])

    # shuffle the data in the arrays
    for c in class_dict:
        random.shuffle(class_dict.get(c))

    # create train and test set
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for c in class_dict:
        amt_of_train_samples = round(0.8 * len(class_dict.get(c)))
        amt_of_test_samples = len(class_dict.get(c)) - amt_of_train_samples

        X_train.extend(class_dict.get(c)[: amt_of_train_samples])
        y_train.extend([c] * amt_of_train_samples)

        X_test.extend(class_dict.get(c)[amt_of_train_samples:])
        y_test.extend([c] * amt_of_test_samples)

    train_set = X_train, y_train
    test_set = X_test, y_test

    return train_set, test_set


def test(log_reg, test_set):
    """
    test a logistic regression model
    - mean accuracy
    - F1 score
    - confusion matrix
    :param log_reg: model
    :param test_set: test data set
    """
    X_test, y_test = test_set

    # compute mean accuracy
    mean_accuracy = log_reg.score(X_test, y_test)
    print('The mean accuracy of the logistic regression is: ' + str(mean_accuracy) + '\n')

    # compute f1 score
    y_pred = log_reg.predict(X_test)
    classes = list(range(4))
    f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=classes, average=None)
    print('The F1 score of the logistic regression is: \n')
    for label in classes:
        print('Label ' + str(label) + ': ' + str(f1[label]))

    # compute confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=classes)
    print('Confusion matrix:')
    print(conf_matrix)
