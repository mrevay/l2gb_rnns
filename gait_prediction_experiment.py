import numpy as np
import torch
import scipy.io as io
import os
import sys

import matplotlib.pyplot as plt
import opt.stochastic_nlsdp as nlsdp
import models.diRNN as diRNN
import models.lstm as lstm

import train as train
import data.load_data as load_data

import multiprocessing
import gp_run_aa as aa

# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)
multiprocessing.set_start_method('spawn', True)


# Returns the results of running model on the data in loader.
def test(model, loader):
    model.eval()

    length = loader.__len__()
    inputs = np.zeros((length,), dtype=np.object)
    outputs = np.zeros((length,), dtype=np.object)
    measured = np.zeros((length,), dtype=np.object)

    SE = np.zeros((length, model.ny))
    NSE = np.zeros((length, model.ny))

    with torch.no_grad():
        for idx, (u, y) in enumerate(loader):
            yest = model(u)
            inputs[idx] = u.numpy()
            outputs[idx] = yest.numpy()
            measured[idx] = y.numpy()

            error = yest[0].numpy() - y[0].numpy()
            mu = y[0].mean(1).numpy()
            N = error.shape[1]
            norm_factor = ((y[0].numpy() - mu[0, None])**2).sum(1)

            SE[idx] = (error ** 2 / N).sum(1) ** (0.5)
            NSE[idx] = ((error ** 2).sum(1) / norm_factor) ** (0.5)

    res = {"inputs": inputs, "outputs": outputs, "measured": measured, "SE": SE, "NSE": NSE}
    return res


def test_and_save_model(name, model, train_loader, val_loader, test_loader, log, params=None):

    nx = model.nx
    layers = model.layers
    path = "./experimental_results/gait_prediction/adversarial_attacks/w{}_l{}/".format(nx, layers)
    file_name = name + '.mat'

    train_stats = test(model, train_loader)
    val_stats = test(model, val_loader)
    test_stats = test(model, test_loader)

    data = {"validation": val_stats, "training": train_stats, "test": test_stats, "nx": model.nx, "nu": model.nu, "ny": model.ny,
            "layers": model.layers, "training_log": log}

    if params is not None:
        data = {**data, **params}

    # Create target Directory if doesn't exist
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory ", path, " Created ")

    io.savemat(path + file_name, data)
    torch.save(model.state_dict(), path + name + ".params")


def fgsa(model, loader, epsilon=0.1):
    model.eval()

    length = loader.__len__()
    inputs = np.zeros((length,), dtype=np.object)
    outputs = np.zeros((length,), dtype=np.object)
    measured = np.zeros((length,), dtype=np.object)

    SE = np.zeros((length, model.ny))
    NSE = np.zeros((length, model.ny))

    for idx, (u, y) in enumerate(loader):

        # Calculate adversarial directions
        # for iter in range(100):
        model.zero_grad()
        u.requires_grad = True
        yest_orig = model(u)
        J = model.criterion(yest_orig, y)
        J.backward()

        with torch.no_grad():
            # Adversarial example
            w = u + epsilon * u.grad.sign()
            yest = model(w)

            inputs[idx] = u.detach().numpy()
            outputs[idx] = yest.numpy()
            measured[idx] = y.numpy()

            error = yest[0].numpy() - y[0].numpy()
            mu = y[0].mean(1).numpy()
            N = error.shape[1]
            norm_factor = ((y[0].numpy() - mu[0, None])**2).sum(1)

            SE[idx] = (error ** 2 / N).sum(1) ** (0.5)
            NSE[idx] = ((error ** 2).sum(1) / norm_factor) ** (0.5)

    res = {"inputs": inputs, "outputs": outputs, "measured": measured, "SE": SE, "NSE": NSE}
    return res


# # Run fast gradient sign attack to maximize loss
# def run_fgsa(name, model, train_loader, val_loader, test_loader, params=None):
#     path = "./experimental_results/gait_prediction/adversarial_attacks/"
#     for epsilon in np.linspace(0.0, 0.5, 20):
#         train_attack = fgsa(model, train_loader, epsilon=epsilon)
#         val_attack = fgsa(model, val_loader, epsilon=epsilon)
#         test_attack = fgsa(model, test_loader, epsilon=epsilon)

#         data = {"validation": val_attack, "training": train_attack, "test": test_attack, "epsilon": epsilon}

#         if not os.path.exists(path):
#             os.mkdir(path)
#             print("Directory ", path, " Created ")

#         io.savemat(path + "aa_eps{:1.3f}_".format(epsilon) + name + ".mat", data)


if __name__ == '__main__':

    this_seed = np.random.randint(0, 2**32)

    mu = 0.05
    eps = 1E-4

    init_var = 1.6
    init_offset = 1E-3  # a small perturbatiuon to ensure strict feasbility of initial point

    max_epochs = 500
    patience = 10

    layers = 2
    subject = 1
    layers = int(sys.argv[1])

    lr_decay = 0.99
    lr = 1E-4

    print("Training models with {} layers".format(layers))
    width = 64
    lstm_width = 49

    for val_set in range(2, 8):
    # Load the data set
        dataset_options = load_data.make_default_options(train_bs=1, train_sl=2048, val_bs=10, ar=False, val_set=val_set)
        dataset_options["subject"] = subject
        train_loader, val_loader, test_loader, scaling_factors = load_data.load_dataset(dataset="gait_prediction_stairs", dataset_options=dataset_options)

        nu = train_loader.nu
        ny = train_loader.ny

        # Options for the solver
        # solver_options = nlsdp.make_stochastic_nlsdp_options(max_epochs=max_epochs, lr=5.0E-4, mu0=100, lr_decay=0.98)
        solver_options = nlsdp.make_stochastic_nlsdp_options(max_epochs=max_epochs, lr=lr, mu0=10, lr_decay=lr_decay, patience=patience)

        # for gamma in [0.5, 1, 1.5, 2.5, 5]:
        for gamma in [0.5, 1, 1.5, 2.5, 5]:
            # Train l2 gain bounded implicit model ------------------------------------------------------------------------
            name = "dl2_gamma{:1.2f}_sub{:d}_val{:d}".format(gamma, subject, val_set)

            print("training model: " + name)
            model = diRNN.diRNN(nu, width, ny, layers, nBatches=9, nl=torch.tanh)

            # Add 0.1 to ensure we are strictly feasible after initialization
            model.init_dl2(epsilon=eps + init_offset, gamma=gamma, init_var=init_var, custom_seed=val_set + this_seed)
            log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                    options=solver_options, LMIs=model.dl2_lmi(gamma=gamma, epsilon=eps))

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log, params=scaling_factors)

        # # Train Contracting model
        # name = "contracting_sub{:d}_val{:d}".format(subject, val_set)
        # model = diRNN.diRNN(nu, width, ny, layers, nBatches=9, nl=torch.tanh)
        # model.init_l2(mu=mu, epsilon=eps + init_offset, init_var=init_var, custom_seed=this_seed + val_set)

        # log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        #                                         options=solver_options, LMIs=model.contraction_lmi(mu=mu, epsilon=eps))

        # test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log, params=scaling_factors)

        # # Train an LSTM network
        # name = "LSTM_sub{:d}_val{:d}".format(subject, val_set)
        # model = lstm.lstm(nu, lstm_width, ny, layers=layers)
        # log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        #                                         options=solver_options)

        # test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log, params=scaling_factors)
