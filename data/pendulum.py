import torch
import matplotlib.pyplot as plt
import numpy as np
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


def random_bit_stream(T, period, u_sd):
    u_per = []
    while(sum(u_per) < T):
        u_per += [int((period * np.random.rand()))]

    # shorten the last element so that the periods add up to the total length
    u_per[-1] = u_per[-1] - (sum(u_per) - T)

    u = np.concatenate([u_sd * (np.random.rand()-0.5) * np.ones((per, 1)) for per in u_per], 0)
    return u


def pendulum(batches=1, T=10000, Ts=0.01, input_period=20, input_mag=2, meas_sd=0.01, proc_sd=0.0, x0=None, input_bias=0.0):

    g = 9.81
    L = 1
    D = 2.5

    time = np.linspace(0, Ts*T, T)
    utild =  np.stack([random_bit_stream(T, input_period, input_mag) for b in range(batches)], 0).transpose(1, 2, 0) + input_bias

    proc_noise = proc_sd * (np.random.rand(T, 1, batches) - 0.5)

    pend_dyn = lambda x, u, w: np.array([[x[0] + Ts * x[1],
                                        x[1] + Ts * (w[0] + 4 * u[0] - g / L * np.sin(x[0]) - D * x[1])]])

    xtild = np.zeros((T, 2, batches))
    if x0 is not None:
        xtild[0] = x0

    for ii in range(1, T):
        xnext = pend_dyn(xtild[ii-1], utild[ii-1], proc_noise[ii-1])
        # xnext[0] = (xnext[0] + np.pi) % (2 * np.pi) - np.pi
        xtild[ii] = xnext

    xtild = xtild.transpose(2, 0, 1)
    utild = utild.transpose(2, 0, 1)

    ytild = xtild + np.random.randn(T, 2)*meas_sd
    # ytild = np.stack([np.cos(xtild[:, :, 0]), np.sin(xtild[:, :, 0]), xtild[:, :, 1]], 2) + np.random.randn(T, 3) * meas_sd

    return time, utild, xtild, ytild

def get_ic_response(x0, T, Ts, meas_sd):
    t, u, x, y = pendulum(1, T, Ts, 100, 0, meas_sd=meas_sd, x0=x0)
    return t, u, x, y

def get_step_response(step_size, T, Ts, meas_sd):
    t, u, x, y = pendulum(1, T, Ts, 100, 0, meas_sd=meas_sd, input_bias=step_size)
    return t, u, x, y

def load_pendulum_data(batches, T, Ts, input_period, input_mag, meas_sd=0.01, mini_batch_size=1, proc_sd=0.0):

    #  Form training data
    t, u, x, y = pendulum(batches, T, Ts, input_period, input_mag, meas_sd=meas_sd, proc_sd=proc_sd)
    # u_batched = u.reshape(batches, -1, 1).transpose((0, 2, 1))
    # y_batched = y.reshape(batches, -1, 3).transpose((0, 2, 1))

    #Normalize the data
    # mean_u = np.mean(u[0], axis=1)
    # mean_y = np.mean(y[0], axis=1)

    # std_u = u[0].std(axis=1)
    # std_y = y[0].std(axis=1)

    # normalize_u = lambda X: (X - mean_u[:, None]) / std_u[:, None]
    # normalize_y = lambda X: (X - mean_y[:, None]) / std_y[:, None]

    # Turn into IO data
    u_split = [u[b].T for b in range(batches)]
    y_split = [y[b].T for b in range(batches)]
    io_dat = IO_data(u_split, y_split)

    # Then into data loaders
    # meas_loader = DataLoader(measured, batch_size=1, shuffle=True, num_workers=1)
    # true_loader = DataLoader(true, batch_size=1, shuffle=True, num_workers=1)

    loader = DataLoader(io_dat, batch_size=mini_batch_size, shuffle=True, num_workers=1)

    loader.nu = u_split[0].shape[0]
    loader.ny = y_split[0].shape[0]

    return loader
