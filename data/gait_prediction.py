import torch
import scipy.io
import urllib
import urllib.request
import zipfile
import shutil
import os
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class IO_data(Dataset):

    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, inputs, outputs, seq_len=None):

        self.nu = inputs[0].shape[0]
        self.ny = outputs[0].shape[0]

        self.nBatches = inputs.__len__()

        if torch.get_default_dtype() is torch.float32:
            convert = lambda x: x.astype(np.float32)
        else:
            convert = lambda x: x.astype(np.float64)

        self.u = list(map(convert, inputs))
        self.y = list(map(convert, outputs))

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        # return self.u[index][None, ...], self.y[index][None, ...]
        return self.u[index], self.y[index]


def load_data(options, dataset="walk", shuffle_training=True, workers=1, subject=1):
    #  check to see if the data set has already been downloaded.

    folder = "./data/gait_prediction/python_data/".format(subject)
    if dataset == "stairs":
        file_name = "sub{:d}_Upstairs_canes_all.mat".format(subject)
    elif dataset == "walk":
        file_name = "sub{:d}_Walk_canes_all.mat".format(subject)

    data = scipy.io.loadmat(folder + file_name)

    # What even is this data format ....
    transpose = lambda x: x.T
    inputs = list(map(transpose, data["p_data"][0, 0][0][0]))
    outputs = list(map(transpose, data["p_data"][0, 0][1][0]))

    # split data into training, validation and test
    L = inputs.__len__()
    val_set = options["val_set"]

    # test sets
    test_u = inputs[-2:]
    test_y = outputs[-2:]

    # val sets
    val_u = [x for i, x in enumerate(inputs) if i == val_set]
    val_y = [x for i, x in enumerate(outputs) if i == val_set]

    # training sets
    train_u = [x for i, x in enumerate(inputs) if i != val_set and i < L - 2]
    train_y = [x for i, x in enumerate(outputs) if i != val_set and i < L - 2]

    #Normalize the data
    mean_u = np.mean(train_u[0], axis=1)
    mean_y = np.mean(train_y[0], axis=1)

    std_u = train_u[0].std(axis=1)
    std_y = train_y[0].std(axis=1)

    sf = {"mean_u": mean_u, "mean_y": mean_y, "std_u": std_u, "std_y": std_y}

    normalize_u = lambda X: (X - mean_u[:, None]) / std_u[:, None]
    normalize_y = lambda X: (X - mean_y[:, None]) / std_y[:, None]

    # normalize inputs
    train_u = list(map(normalize_u, train_u))
    val_u = list(map(normalize_u, val_u))
    test_u = list(map(normalize_u, test_u))

    # normalize outputs
    train_y = list(map(normalize_y, train_y))
    val_y = list(map(normalize_y, val_y))
    test_y = list(map(normalize_y, test_y))

    training = IO_data(train_u, train_y)
    validation = IO_data(val_u, val_y)
    test = IO_data(test_u, test_y)

    # training = IO_data(inputs[0:-2], outputs[0:-2])
    # validation = IO_data(inputs[-2:], outputs[-2:-1])
    # test = IO_data(inputs[-2:], outputs[-2:])

    # Merge the list of arrays into a 3D tensor - just kdding, the trials have different lengths
    # merge_arrays = lambda x, y: np.concatenate((x[None, ...], y[None, ...]), 0)
    # inputs = reduce(merge_arrays, inputs)
    # outputs = reduce(merge_arrays, outputs)

    train_loader = DataLoader(training, batch_size=1, shuffle=shuffle_training, num_workers=workers)
    val_loader = DataLoader(validation, batch_size=1, num_workers=workers)
    test_loader = DataLoader(test, batch_size=1, num_workers=workers)

    train_loader.nu = training.nu
    train_loader.ny = training.ny

    val_loader.nu = training.nu
    val_loader.ny = training.ny

    test_loader.nu = training.nu
    test_loader.ny = training.ny

    return train_loader, val_loader, test_loader, sf
