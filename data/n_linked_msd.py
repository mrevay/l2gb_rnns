import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interp
from torch.utils.data import Dataset, DataLoader


class IO_data(Dataset):

    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, U, X0, Xnext):

        self.nu = U.shape[1]
        self.nx = X0.shape[1]

        self.nBatches = U.shape[0]

        if torch.get_default_dtype() is torch.float32:
            convert = lambda x: x.astype(np.float32)
        else:
            convert = lambda x: x.astype(np.float64)

        self.u = convert(U)
        self.X0 = convert(X0)
        self.Xnext = convert(Xnext)

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        # return self.u[index][None, ...], self.y[index][None, ...]
        return [self.X0[index, :], self.u[index, :]], self.Xnext[index, :]


class sim_IO_data(Dataset):
    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, U, X):

        self.nu = U.shape[1]
        self.nx = X.shape[1]

        self.nBatches = 1

        if torch.get_default_dtype() is torch.float32:
            convert = lambda x: x.astype(np.float32)
        else:
            convert = lambda x: x.astype(np.float64)

        self.u = convert(U)
        self.X0 = convert(X[:, 0])
        self.X = convert(X)

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        return [self.X0, self.u], self.X


class msd_chain():
    def __init__(self, N=5, T=100, Ts=0.1, u_sd=1, period=50, batchsize=1):
        # self.k = np.random.rand(N)
        # self.c = 1 * np.random.rand(N)
        # self.m = 1 + np.random.rand(N)

        self.k = np.linspace(1.0, 2.0, N)
        self.c = 5*np.linspace(0.5, 0.5, N)
        self.m = np.linspace(1, 2, N)

        self.N = N
        self.u_sd = u_sd
        self.period = period
        self.travel = 2.5

        # number of samples
        self.T = T

        # Sampling period
        self.Ts = Ts

        self.batchsize = batchsize

    #  Generate a piecewise constant input with randomly varying period and magnitude
    def random_bit_stream(self, T=None):
        if T is None:
            T = self.T

        u_per = []
        while(sum(u_per) < T):
            u_per += [int((self.period * np.random.rand()))]

        # shorten the last element so that the periods add up to the total length
        u_per[-1] = u_per[-1] - (sum(u_per) - T)

        u = np.concatenate([self.u_sd * (np.random.rand() - 0.5) * np.ones((per, 1)) for per in u_per], 0)
        return u

    # For displacements x, returns the spring force
    def spring_func(self, x):
        epsilon = 1E-5
        d = np.clip(x, -self.travel, self.travel)
        return np.tan(np.pi * d / 2 / (self.travel + epsilon))

    def dynamics(self, x, u):
        d = x[0::2, :]
        v = x[1::2, :]

        # Vector containing the forces on each cart
        F = np.zeros_like(d)

        # Force on first cart first cart
        F[0] += (u[0] + self.k[0] * self.spring_func(-d[0])
                 + self.k[1] * self.spring_func(d[1] - d[0])
                 - self.c[0] * v[0]
                 + self.c[1] * (v[1] - v[0]))

        # Force on the middle carts
        F[1:-1] += (self.k[1:-1, None] * self.spring_func(d[0:-2, :] - d[1:-1, :])
                    + self.k[2:, None] * self.spring_func(d[2:, :] - d[1:-1, :])
                    + self.c[1:-1, None] * (v[0:-2, :] - v[1:-1, :])
                    + self.c[2:, None] * (v[2:, :] - v[1:-1, :]))

        # Force on the last cart
        F[-1] = (self.k[-1, None] * self.spring_func(d[-2, :] - d[-1, :])
                 + self.c[-1, None] * (v[-2, :] - v[-1, :]))

        dxdt = np.zeros_like(x)
        dxdt[0::2] = v
        dxdt[1::2] = F / self.m[:, None]

        return dxdt

    def simulate(self):
        # End time of simulation
        Tend = self.T * self.Ts
        time = np.linspace(0, Tend, self.T)

        # input and function to sample input
        u = self.random_bit_stream()
        u_interp = interp.interp1d(time, u[:, 0])

        # Construct function for the dynamcis of the system
        # dyn = lambda t, x: self.dynamics(x, u_interp(t))
        x0 = np.zeros((2 * self.N))
        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t)[None])
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], x0, t_eval=time)
        t = sol['t']
        x = sol['y']
        u = u.T

        data = sim_IO_data(u, x)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
        return loader

    # Grid the states space X x U with res points in all directions
    def grid_ss(self, res):
        # End time of simulation
        samples = 10
        Tend = samples * self.Ts
        time = np.linspace(0, samples * self.Ts, 10)

        # grid over control input
        u = [np.linspace(-1, 1, res)]
        # grid for each state
        x = [np.linspace(-1, 1, res) for n in range(2 * self.N)]

        X = np.meshgrid(*(x + u))  # unpack list into meshgrid

        U = X[-1].reshape(1, -1)
        X0 = np.stack([x.reshape(-1) for x in X[:-1]]).reshape(-1)

        # Construct function for the dynamcis of the system
        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, U)
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], X0, t_eval=time)
        t = sol['t']
        Y = sol['y'].reshape(2 * self.N, -1).T
        X = X0.reshape(2 * self.N, -1).T
        U = U.T

        data = IO_data(U, X, Y)

        # convert to data loader
        loader = DataLoader(data, batch_size=self.batchsize, shuffle=True, num_workers=4)
        return loader

    def sim_ee(self, T=10000):
        # End time of simulation
        Tend = T * self.Ts
        time = np.linspace(0, Tend, T)

        # input and function to sample input
        u = self.random_bit_stream(T)
        u_interp = interp.interp1d(time, u[:, 0])

        # Construct function for the dynamcis of the system
        # dyn = lambda t, x: self.dynamics(x, u_interp(t))
        x0 = np.zeros((2 * self.N))
        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t)[None])
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], x0, t_eval=time)
        t = sol['t']
        x = sol['y']
        u = u.T

        U = u[:, :-1].T
        X = x[:, :-1].T
        Y = x[:, 1:].T

        data = IO_data(U, X, Y)
        loader = DataLoader(data, batch_size=self.batchsize, shuffle=False, num_workers=1)
        return loader
