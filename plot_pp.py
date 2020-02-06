import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import numpy as np
import scipy as sp
import scipy.io as io

import data.pendulum as pendulum

import time, threading

import  models.diRNN as diRNN
import models.lstm as lstm


# Callback to automatically refresh plot  every dt seconds
def refresh_fig(dt=1 / 20):
    plt.pause(0.0001)
    threading.Timer(dt, refresh_fig).start()


def plot_trajectory(y):
    for b in range(y.shape[0]):
        plt.plot(y[b, 0].detach().numpy(), y[b, 1].detach())


def get_ic_response(model, y0, batches=100, T=1000):
    # theta0 = np.linspace(-0., 1, N)
    # dtheta0 = np.linspace(-1.5, 1.5, N)

        # y0 = np.array([[theta0[idx]], [np.pi]])

        # y0 = np.array([[-0.5], [dtheta0[idx]]])

        # Calculate a series of inital states that give the outputs in y0
    C = model.output_layer.weight.detach().numpy()
    b = model.output_layer.bias.detach().numpy()
    b = b[:, None]

    Cinv = np.linalg.pinv(C)
    Cnull = sp.linalg.null_space(C)

    # First find least squares solution using pseudo-inverse
    h0_star = Cinv @ (y0 - b)
    alphas = 1 * (np.random.rand(Cnull.shape[1], batches) - 0.5)  # Distribute initial states over unit cube and see how they perform

    h0 = h0_star + Cnull @ alphas

    # Simulate model on these
    u = torch.zeros((batches, 1, T))
    y = model(u, torch.Tensor(h0.T))

    return u, y

    print("Just plotted")

def plot_val(name, width=64, lstm_width=49, nu=1, ny=2, layers=2, train_batches=50, nl=None, T=1000, plot_str='k', x0=None, LSTM=False):

    path = "./experimental_results/pendulum/"
    non_lin = torch.relu if nl is None else nl

    data = io.loadmat(path + name + ".mat")

    if LSTM:
        model = lstm.lstm(nu, lstm_width, ny, layers)
        model.load_state_dict(torch.load(path + "p_" + name))
        model.output_layer = model.output
    else:
        model = diRNN.diRNN(nu, width, ny, layers, nBatches=train_batches, nl=non_lin, learn_init_state=False)
        model.load_state_dict(torch.load(path + "p_" + name))

    u, yest = get_ic_response(model, x0, T=T, batches=1000)
    plt.plot(yest[:, 0].detach().numpy().T, yest[:, 1].detach().numpy().T, plot_str)
    plt.pause(0.01)


def plot_sim_step_response(step_size=1.0, T=500, Ts=0.1, meas_sd=0.01):
    t, u, x, y = pendulum.get_step_response(T=T, Ts=Ts, meas_sd=meas_sd, step_size=step_size)
    plt.plot(y[0, :, 0], y[0, :, 1], 'k')
    plt.pause(0.01)

def plot_step_response(model, step_size=1.0, T=500, Ts=0.1, plot_args=None):

    if plot_args is None:
        plot_args = {"linestyle": '-'}

    init_period = 100

    u = step_size * torch.ones((1, 1, T))
    u[0, 0, 0:init_period] = 0.0
    y = model(u)
    plt.plot(y[0, 0, init_period:].detach(), y[0, 1, init_period:].detach(), **plot_args)
    plt.pause(0.01)

def plot_lstm_step_response(model, step_size=1.0, T=500, Ts=0.1, plot_args=None):

    if plot_args is None:
        plot_args = {"linestyle": '-'}

    init_period = 100

    u = step_size * torch.ones((1, 1, T))
    u[0, 0, 0:init_period] = 0.0
    h0 = 1000 * (torch.rand(2, 1, model.nx) - 0.5)
    c0 = 1000 * (torch.rand(2, 1, model.nx) - 0.5)
    y = model(u, h0=h0, c0=c0)
    plt.plot(y[0, 0, init_period:].detach(), y[0, 1, init_period:].detach(), **plot_args)
    plt.pause(0.01)

if __name__ == "__main__":


    # Must Knowd widths of the networks being loaded
    width = 64
    lstm_width = 49
    nu = 1
    ny = 2
    layers = 2
    train_batches = 50

    T = 1000

    f2 = plt.figure()

    ss  = 0.5
    plot_sim_step_response(step_size=ss, T=T, Ts=0.1, meas_sd=0.01)

    gammas = [0.2, 0.5, 1.0, 5.0, 10.0, 50.0]
    colors = cm.hot(np.linspace(0.25, 0.75, gammas.__len__()))

    for val_set in range(1, 9):

        for (cid, gamma) in enumerate(gammas):
            color = colors[cid]
            plot_args = {"color": color, "linestyle": '-'}
            model_c = diRNN.diRNN(nu, width, ny, layers, nBatches=train_batches, nl=torch.relu, learn_init_state=False)
            model_c.load_state_dict(torch.load("./experimental_results/pendulum/p_dl2_gamma{:1.1f}_val{:d}".format(gamma, val_set)))
            plot_step_response(model_c, step_size=ss, T=T, plot_args=plot_args)


        plot_args = {"color": 'g', "linestyle": '-'}
        model_c = diRNN.diRNN(nu, width, ny, layers, nBatches=train_batches, nl=torch.relu, learn_init_state=False)
        model_c.load_state_dict(torch.load("./experimental_results/pendulum/p_contracting_val{:d}".format(val_set)))
        plot_step_response(model_c, step_size=ss, T=T, plot_args=plot_args)


        plot_args = {"color": 'b', "linestyle": '-'}
        model_lstm = lstm.lstm(nu, lstm_width, ny, layers=layers)
        model_lstm.load_state_dict(torch.load("./experimental_results/pendulum/p_LSTM_val{:d}".format(val_set)))
        model_lstm.eval()
        model_lstm.output_layer = model_lstm.output

        for b in range(20):
            plot_lstm_step_response(model_lstm, step_size=ss, T=T, plot_args=plot_args)

    print("fin")
