import torch
import numpy as np
import matplotlib.pyplot as plt
import models.lstm as lstm
import models.diRNN as diRNN
import data.load_data as load_data
from scipy.optimize import minimize, NonlinearConstraint
import multiprocessing
import scipy.io as io

import sys



# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)
multiprocessing.set_start_method('spawn', True)


def run_aa(model, loader, epsilon=0.1, alpha=50):
    model.eval()
    length = loader.__len__()
    inputs = np.zeros((length,), dtype=np.object)
    outputs1 = np.zeros((length,), dtype=np.object)
    outputs2 = np.zeros((length,), dtype=np.object)
    measured = np.zeros((length,), dtype=np.object)

    Delta_u = np.zeros((length,), dtype=np.object)
    dydu = []

    SE = np.zeros((length, model.ny))
    NSE = np.zeros((length, model.ny))

    for idx, (u, y) in enumerate(loader):

        # Calculate adversarial directions
        def objective(delta_u):
            with torch.no_grad():
                du = np.reshape(delta_u, u.shape)
                yest = model(u + torch.Tensor(du))
            return -np.linalg.norm(yest.detach().numpy() - y.detach().numpy())

        def Jacobian(delta_u):
            model.zero_grad()
            du = torch.Tensor(np.reshape(delta_u, u.shape))
            yest = model(u + du)
            J = -np.linalg.norm(yest - y)
            J.backward()

            return du.grad.reshape(-1).detach().numpy()

        def constraint(delta_u):
            return min(epsilon ** 2 - delta_u.norm() ** 2, 0)

        J_last = -1
        # decision vairable
        delta_u = torch.zeros_like(u, requires_grad=True)
        for ii in range(10000):
            # zero gadients
            model.zero_grad()
            if delta_u.grad is not None:
                delta_u.grad.zero_()

            # calculate objective with
            yest1 = model(u)
            yest = model(u + delta_u)
            J = -model.criterion(yest, y)
            J.backward()
            # print(model.criterion(yest, y))

            # gradient descent step
            # if delta_u.grad is not None:
            delta_u.data -= alpha * delta_u.grad

            # project back onto norm ball
            if delta_u.data.norm() / u.norm() > epsilon:
                delta_u.data = delta_u.data * epsilon / delta_u.data.norm() * u.norm()
                alpha *= 0.95

            if abs(J - J_last) < 1E-4:
                break

            sys.stdout.write("\tdJ = {:2.4f}\r".format(float(J - J_last)))
            J_last = float(J)
        print()
        # Calculate the statistics of interest
        yest1 = model(u)
        yest2 = model(u + delta_u)
        du = delta_u[0].detach().numpy()
        dy = (yest2 - yest1)[0].detach().numpy()

        if np.linalg.norm(du, ord="fro") == 0.0:
            dydu += [0]
        else:
            dydu += [np.linalg.norm(dy, ord="fro") / np.linalg.norm(du, ord="fro")]

        # model.check_storage(u, u + delta_u, gamma=gamma)

        inputs[idx] = u.detach().numpy()
        outputs1[idx] = yest1.detach().numpy()
        outputs2[idx] = yest2.detach().numpy()
        measured[idx] = y.detach().numpy()

        Delta_u[idx] = du

        error = yest2[0].detach().numpy() - y[0].detach().numpy()
        mu = y[0].mean(1).detach().numpy()
        N = error.shape[1]
        norm_factor = ((y[0].detach().numpy() - mu[0, None])**2).sum(1)

        SE[idx] = (error ** 2 / N).sum(1) ** (0.5)
        NSE[idx] = ((error ** 2).sum(1) / norm_factor) ** (0.5)

    results = {"dydu": dydu, "u": inputs, "du": Delta_u, "y": outputs1,
               "yp": outputs2, "NSE": NSE, "SE": SE, "epsilon": epsilon}

    return results


def save_aa(path, name, model, test_loader, train_loader, alpha=50):
    print("Running aa with model: " + name)
    for epsilon in np.linspace(0, 0.2, 11):
        results_test = run_aa(model, test_loader, epsilon=epsilon, alpha=alpha)
        # results_train = run_aa(model, train_loader, epsilon=epsilon)
        print("\tepsilon: {:1.2f}, \t|dy|/|du| = {:1.2f}".format(epsilon, results_test["dydu"][0]))

        io.savemat(path + "test_aa_eps{:1.2f}_".format(epsilon) + name + ".mat", results_test)
        # io.savemat(path + "train_aa_eps{:1.2f}_".format(epsilon) + name + ".mat", results_train)


if __name__ == "__main__":

    mu = 0.05
    eps = 1E-3

    init_var = 1.2
    init_offset = 0.05  # a small perturbatiuon to ensure strict feasbility of initial point

    max_epochs = 500
    patience = 20

    layers = int(sys.argv[1])
    subject = 1
    val_set = 1

    lr_decay = 0.99
    lr = 1E-3

    print("Testing models with {} layers".format(layers))
    width = 64
    lstm_width = 49

    nu = 14
    ny = 6

    # Load the datasets
    dataset_options = load_data.make_default_options(train_bs=1, train_sl=2048, val_bs=10, ar=False, val_set=val_set)
    dataset_options["subject"] = subject
    train_loader, val_loader, test_loader, scaling_factors = load_data.load_dataset(dataset="gait_prediction_stairs", dataset_options=dataset_options)

    for val_set in range(2, 9):

    # Train l2 gain bounded implicit model ------------------------------------------------------------------------
        name = "LSTM_sub{:d}_val{:d}".format(subject, val_set)
        path = "./experimental_results/gait_prediction/adversarial_attacks/w49_l{:d}/".format(layers)

        model = lstm.lstm(nu, lstm_width, ny, layers=layers)
        model.load_state_dict(torch.load(path + name + ".params"))
        save_aa(path, name, model, test_loader, train_loader, alpha=50)

    # Contracting implicit model ------------------------------------------------------------------------
        name = "contracting_sub{:d}_val{:d}".format(subject, val_set)
        path = "./experimental_results/gait_prediction/adversarial_attacks/w64_l{:d}/".format(layers)

        model = diRNN.diRNN(nu, width, ny, layers, nBatches=9, nl=torch.tanh)
        model.load_state_dict(torch.load(path + name + ".params"))
        save_aa(path, name, model, test_loader, train_loader, alpha=200)

        # run adversarial attacks for varying gamma
        for gamma in [0.5, 1, 1.5, 2.5, 5.0]:
            # Train l2 gain bounded implicit model ------------------------------------------------------------------------
            name = "dl2_gamma{:1.2f}_sub{:d}_val{:d}".format(gamma, subject, val_set)
            path = "./experimental_results/gait_prediction/adversarial_attacks/w64_l{:d}/".format(layers)

            print("Loading Model: ", name)

            model = diRNN.diRNN(nu, width, ny, layers, nBatches=9, nl=torch.tanh)
            model.load_state_dict(torch.load(path + name + ".params"))

            # Check that the LMIs hold
            print("Checking that model satisfies conditions")
            eps = 1E-5
            LMIs = model.dl2_lmi(gamma=gamma, epsilon=eps)
            for (idx, lmi) in enumerate(LMIs):
                eigs = lmi().symeig()[0]
                if all(eigs >= 0.0):
                    print("\tLMI ", idx, "checked: Okay!")
                else:
                    print("\tLMI", idx, "not positive definite")

            for layer in range(layers):
                if all(model.E[layer].symeig()[0] > 1E-8):
                    print("\tE[", layer, "] >0 checked: okay!")
                else:
                    print("\tE[", layer, "] is n.d.")

                if all(model.P[layer] > 1E-8):
                    print("\tP[", layer, "] >0 checked: okay!")
                else:
                    print("\tP[", layer, "] is n.d.")

            save_aa(path, name, model, test_loader, train_loader, alpha=200)

