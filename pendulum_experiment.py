import numpy as np
import torch
import scipy.io as io
import sys as sys
import os

# import opt.stochastic_nlsdp as nlsdp
import opt.snlsdp_ipm as nlsdp
# import models.diRNN as diRNN
import models.diRNN as diRNN
import models.dnb as dnb
import models.lstm as lstm

import train as train
import data.pendulum as pendulum

import multiprocessing
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
    path = "./experimental_results/pendulum/".format(nx, layers)
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


if __name__ == '__main__':


    # parameters for the pendulum simulatiuon
    input_mag = 4
    period = 100
    sample_rate = 0.1
    samples = 200
    # train_batches = samples // period // 2
    train_batches = 20
    val_batches = 1
    test_batches = 1

    mini_batch_size = 5

    mu = 0.0001
    eps = 1E-3

    init_var = 2.0
    init_offset = 0.05  # a small perturbation to ensure strict feasbility of initial point

    max_epochs = 200
    patience = 200

    lr_decay = 0.97
    lr = 1E-3

    width = 124
    lstm_width = 110
    layers = 2
    meas_sd = 0.05
    proc_sd = 0.5

    print("Training models with {} layers".format(layers))

    for val_set in range(1, 9):

        # Load the data set
        train_loader = pendulum.load_pendulum_data(train_batches, samples, sample_rate, period,
                                                   input_mag, meas_sd=meas_sd, proc_sd=proc_sd,mini_batch_size=mini_batch_size)
        val_loader = pendulum.load_pendulum_data(val_batches, 5000, sample_rate, period, input_mag, meas_sd=meas_sd, proc_sd=proc_sd)
        test_loader = pendulum.load_pendulum_data(test_batches, 10000, sample_rate, period, input_mag, meas_sd=meas_sd, proc_sd=proc_sd)

        nu = train_loader.nu
        ny = train_loader.ny

        # Options for the solver
        # solver_options = nlsdp.make_stochastic_nlsdp_options(max_epochs=max_epochs, lr=5.0E-4, mu0=100, lr_decay=0.98)
        solver_options = nlsdp.make_stochastic_nlsdp_options(max_epochs=max_epochs, lr=lr, mu0=100,
                                                             lr_decay=lr_decay, patience=patience, clip_at=0.2)

        # Train Contracting model
        name = "contracting_val{:d}".format(val_set)
        model = diRNN.diRNN(nu, width, ny, layers, nBatches=train_batches, nl=torch.relu, learn_init_state=False)
        model.init_l2(mu=mu, epsilon=eps + init_offset, init_var=init_var)

        log, best_model, model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                       options=solver_options, LMIs=model.contraction_lmi(mu=mu, epsilon=eps))

        test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

        path = "./experimental_results/pendulum/" + "bp_" + name
        torch.save(best_model.state_dict(), path)

        path = "./experimental_results/pendulum/" + "p_" + name
        torch.save(model.state_dict(), path)

        # # Train an LSTM network
        # name = "LSTM_val{:d}".format(val_set)
        # model = lstm.lstm(nu, lstm_width, ny, layers=layers, batches=train_batches, learn_init_hidden=False)
        # log, best_model, model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        #                                                options=solver_options)

        # test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

        # path = "./experimental_results/pendulum/" + "bp_" + name
        # torch.save(best_model.state_dict(), path)

        # path = "./experimental_results/pendulum/" + "p_" + name
        # torch.save(model.state_dict(), path)

        for gamma in [0.2, 0.5, 1, 5, 10, 50]:
            # Train l2 gain bounded implicit model ------------------------------------------------------------------------
            name = "dl2_gamma{:1.1f}_val{:d}".format(gamma, val_set)
            model = diRNN.diRNN(nu, width, ny, layers, nBatches=train_batches, nl=torch.relu)

            # Add 0.1 to ensure we are strictly feasible after initialization
            model.init_dl2(epsilon=eps + init_offset, gamma=gamma, init_var=init_var)
            log, best_model, model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                           options=solver_options, LMIs=model.dl2_lmi(gamma=gamma, epsilon=eps))

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

            path = "./experimental_results/pendulum/" + "bp_" + name
            torch.save(best_model.state_dict(), path)

            path = "./experimental_results/pendulum/" + "p_" + name
            torch.save(model.state_dict(), path)
