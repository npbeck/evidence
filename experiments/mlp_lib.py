"""
author: Nils Beck, nils.beck@pm.me

Evidence Project
@ Ubiquitous Knowledge Processing Lab
@ TU Darmstadt, Germany

May 2020

In this file I provide all functionality necessary for training a multilayer perceptron (MLP)
 - a simple neural network.
The model will read sentences (or rather sentence embeddings) and compute a score, representing how well
they are suited as example sentences for a dictionary
"""
import numpy
import torch
import torch.optim.adagrad
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

# variables that need to be adjusted according to your storage system
trained_model_location = '/ukp-storage-1/nbeck/sentence-bert-venv/mlp_model.pth'

# variables depending on the data
input_size = 512  # length of input vectors
classes = ('0', '1', '2', '3')
amt_of_classes = len(classes)  # amount of different classes


class EvidenceDataSet(Dataset):
    """
    A data set
    We use this format since it works well with the PyTorch Libraries
    """

    def __init__(self, embeddings_data, labels_data):
        """
        Initialize a data set
        :param embeddings_data: path to numpy file containing the respective embeddings
        :param labels_data: path to numpy file containing the respective labels
        """
        self.embeddings = embeddings_data
        self.labels = labels_data
        assert (len(self.embeddings) == len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        returns the i-th element in the data set
        :param index: index of the element
        :return: embeddings in tensor format and corresponding label as int
        """
        return torch.from_numpy(self.embeddings[index]), self.labels[index]


class MLP(nn.Module):
    """
    A Multilayer Perceptron
    """

    def __init__(self):
        super(MLP, self).__init__()
        # we will create three layers in total, i.e. two hidden layers
        self.fc1 = nn.Linear(input_size, 256, True)
        self.fc2 = nn.Linear(256, 64, True)
        self.fc3 = nn.Linear(64, amt_of_classes, True)

    def forward(self, x):
        """
        perform forward propagation through the network
        :param x: an input sample
        :return: the result of the input sample being propagated through the network
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


def train(model: MLP, train_set: EvidenceDataSet, dev_set: EvidenceDataSet, amt_of_epochs: int):
    """
    train the given MLP model
    :param dev_set: development data set
    :param amt_of_epochs: amount of training epochs, i.e. times that we loop over the same training set
    :param model: a MLP model
    :param train_set: the training data
    :return: the trained MLP model
    """
    # set model's mode to training mode
    model.train(True)

    # load data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                               shuffle=True, num_workers=2)

    # define a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters())

    interim_results = []    # to store the dev results
    for epoch in range(amt_of_epochs):  # loop over the data set multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            """ TODO evaluate the current model using the dev set 
                if it is better than the previous model overwrite it, else perform next epoch with previous model
            """
            interim_results.append(tuple((model, test(model, dev_set))))  # store current epoch's model and its accuracy

    print('Finished Training')
    return interim_results.sort(key=lambda tup: tup[1], reverse=True)[0][0]  # return best-performing model


def test(model: MLP, test_set: EvidenceDataSet):
    """
    test a trained MLP model on test data
    :param model: a MLP model
    :param test_set: the test data set
    :return: the overall accuracy of the model, measured on the given set
    """
    # set model's mode to evaluation mode
    model.eval()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            embeddings, labels = data
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    overall_accuracy = correct / total
    print('Accuracy of the MLP: %d %%' % (100 * overall_accuracy))

    # evaluate individual classes
    class_correct = list(0. for i in range(amt_of_classes))
    class_total = list(0. for i in range(amt_of_classes))
    with torch.no_grad():
        for data in test_loader:
            embeddings, labels = data
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(amt_of_classes):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    return overall_accuracy


def partition_data(X, y, train_ratio, dev_ratio):
    """
    partition the given lists, containing embeddings and labels respectively, into
    - train set
    - dev set
    - test set
    :param train_ratio: training set ratio on whole data set
    :param dev_ratio:
    :param X: embeddings list
    :param y: labels list
    :return: three EvidenceDataSets for train, dev, and test data
    """

    assert (train_ratio + dev_ratio <= 1)
    assert (len(X) == len(y))

    # shuffle the whole set
    zip_list = list(zip(X, y))
    numpy.random.shuffle(zip_list)
    X, y = zip(*zip_list)

    total_len = len(X)
    train_len = round(total_len * train_ratio)
    dev_len = round(total_len * dev_ratio)

    a = train_len
    b = train_len + dev_len
    train_set = EvidenceDataSet(X[:a], y[:a])
    dev_set = EvidenceDataSet(X[a:b], y[a:b])
    test_set = EvidenceDataSet(X[b:], y[b:])

    return train_set, dev_set, test_set
