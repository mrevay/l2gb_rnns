import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.clip_grad as clip_grad
import gp_run_aa as aa

def is_legal(v):
    legal = not torch.isnan(v).any() and not torch.isinf(v)
    return legal


def plot_response(y, yest):
    plt.cla()
    for batch in range(y.size(0)):
        dt = y.size(2)
        t = np.arange(batch * dt, (batch + 1) * dt)
        plt.plot(t, y[batch].T.detach().numpy(), 'k')
        plt.plot(t, yest[batch].T.detach().numpy(), 'r')
    plt.show()
    plt.pause(0.01)


def make_stochastic_nlsdp_options(max_epochs=100, lr=1E-3, lr_decay=0.95, mu0=10, patience=20, clip_at=0.5):
    options = {"max_epochs": max_epochs, "lr": lr, "tolerance_constraint": 1E-6, "debug": False,
               "patience": patience, "omega0": 1E-2, "eta0": 1E-2, "mu0": mu0, "lr_decay": lr_decay, "clip_at": clip_at
               }
    return options


class stochastic_nlsdp():
    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    def __init__(self, model, train_loader, val_loader, criterion=None, equ=None, max_epochs=1000, lr=1.0, max_ls=50,
                 tolerance_grad=1E-6, tolerance_change=1E-6, tolerance_constraint=1E-6, debug=False,
                 patience=10, omega0=1E-2, eta0=1E-1, mu0=10, lr_decay=0.95, clip_at=1.0):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Add model parameters to list of decision variables
        self.decVars = list(self.model.parameters())

        self.criterion = criterion
        self.patience = patience
        self.lr = lr
        self.lr_decay = lr_decay
        self.max_ls = max_ls

        self.omega0 = omega0
        self.eta0 = eta0
        self.mu0 = mu0
        self.clip_at = clip_at

        if equ is None:
            self.equConstraints = []
        else:
            self.equConstraints = equ

        self.max_epochs = max_epochs
        self.tolerance_constraint = tolerance_constraint

        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad

        self.LMIs = []
        self.regularizers = []

    # Evaluates the equality constraints c_i(x) as a vector c(x)
    def ceq(self):
        if self.equConstraints.__len__() == 0:
            return None

        views = []
        for c in self.equConstraints:
            views.append(c())

        return torch.cat(views, 0)

    # returns a flattened tensor of all parameters
    def flatten_params(self):
        views = []
        for p in self.decVars:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # Returns the gradients as a flattened tensor
    def flatten_grad(self):
        views = []
        for p in self.decVars:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # Adds the SDP constraint Qf > 0. Qf should be a function that returns the LMI
    # so that we want Qf()>0
    def addSDPconstraint(self, Qf):
        self.LMIs += [Qf]

    def eval_LMIs(self):
        if self.LMIs.__len__() == 0:
            return None

        else:
            LMIs = []
            for lmi in self.LMIs:
                LMIs.append(lmi())

            return LMIs

    def checkLMIs(self):
        lbs = []
        for lmi in self.LMIs:
            min_eval = lmi().eig()[0]
            lbs += [min_eval[0].min()]

        return lbs

    # Adds a regualizers reg where reg returns term that we woule like to regularize by
    def add_regularizer(self, reg):
        self.regularizers += [reg]

    # Evaluates the regularizers
    def eval_regularizers(self):

        if self.regularizers.__len__() == 0:
            return None

        res = 0
        for reg in self.regularizers:
            res += reg()

        return res

    #  eta tol is the desired tolerance in the constraint satisfaction
    #  omega_tol is the tolerance for first order optimality
    def solve(self):
        # linsearch parameter

        def validate(loader):
            total_loss = 0.0
            total_batches = 0

            self.model.eval()
            with torch.no_grad():
                for u, y in loader:
                    yest = self.model(u)
                    total_loss += self.criterion(yest, y) * u.size(0)
                    total_batches += u.size(0)

                    # plt.cla()
                    # plt.plot(yest[0].detach().numpy().T)
                    # plt.plot(y[0].detach().numpy().T, 'k')
                    # plt.pause(0.001)

            return float(np.sqrt(total_loss / total_batches))

        # Initial Parameters
        # muk = self.mu0  # barrier parameter
        muk = self.mu0  # barrier parameter

        no_decrease_counter = 0

        with torch.no_grad():
            vloss = validate(self.val_loader)
            tloss = validate(self.train_loader)

            best_loss = vloss
            best_model = self.model.clone()

        log = {"val": [vloss], "training": [tloss], "epoch": [0]}
        optimizer = torch.optim.Adam(params=self.decVars, lr=self.lr)
        # optimizer = torch.optim.Rprop(params=self.decVars, lr=self.lr)

        #  Main Loop of Optimizer
        for epoch in range(self.max_epochs):

            #  --------------- Training Step ---------------
            train_loss = 0.0
            total_batches = 0
            self.model.train()
            for idx, (u, y) in enumerate(self.train_loader):

                def AugmentedLagrangian():
                    optimizer.zero_grad()

                    # h0 = self.model.h0[idx: idx + y.size(0)]
                    yest = self.model(u)
                    L = self.criterion(y, yest)

                    # plt.cla()
                    # plt.plot(y[0].detach().numpy().T, 'k')
                    # plt.plot(yest[0].detach().numpy().T)
                    # plt.pause(0.01)

                    train_loss = float(L) * u.size(0)

                    reg = self.eval_regularizers()
                    if reg is not None:
                        L += reg

                    barrier = 0
                    LMIs = self.eval_LMIs()
                    if LMIs is not None:
                        for lmi in LMIs:
                            barrier += -lmi.logdet() / muk

                            try:
                                _ = torch.cholesky(lmi)  # try a cholesky factorization to ensure positive definiteness
                            except:
                                barrier = torch.tensor(float("inf"))

                    L += barrier

                    L.backward()

                    clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_at, 1)
                    # max_grad = max([torch.norm(p, "inf") for p in self.model.parameters()])
                    g_inf = [p.grad.abs().max() for p in filter(lambda p: p.grad is not None, self.model.parameters())]

                    return L, train_loss, barrier, max(g_inf)

                # Store old parameters
                old_theta = self.model.flatten_params().detach()

                # step model
                Lag, t_loss, barrier, max_grad = optimizer.step(AugmentedLagrangian)
                new_theta = self.model.flatten_params().detach()

                # Perform a backtracking linesearch to avoid inf or NaNs
                alpha = 0.85
                ls = 100
                Lag, t_loss, barrier, _ = AugmentedLagrangian()
                while not is_legal(Lag):

                    # step back by half
                    new_theta = alpha * old_theta + (1 - alpha) * new_theta
                    self.model.write_flat_params(new_theta)

                    ls -= 1
                    if ls == 0:
                        print("maximum ls reached")
                        log["success"] = False
                        return log, best_model

                    Lag, t_loss, barrier, _ = AugmentedLagrangian()
                    print("reducing step size: t_loss = {:.5f}, \t barrier = {:.5f}".format(t_loss, barrier))

                train_loss += t_loss
                total_batches += u.size(0)

                print("Epoch {:4d}: \t[{:04d}/{:04d}],\tlr = {:1.2e},\t avg loss: {:.5f}, Augmented Loss: {:.5f}, |g|: {:.5f}".format(epoch,
                      idx + 1, len(self.train_loader), optimizer.param_groups[0]["lr"], train_loss / total_batches, barrier, max_grad))

            # Reduce learning rate slightly after each epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.lr_decay

            muk = 1.5 * muk if muk < 1E4 else 1E4

            # ---------------- Validation Step ---------------
            vloss = validate(self.val_loader)
            tloss = validate(self.train_loader)

            if vloss < best_loss:
                no_decrease_counter = 0
                best_loss = vloss
                best_model = self.model.clone()

            else:
                no_decrease_counter += 1

            log["val"] += [vloss]
            log["training"] += [tloss]
            log["epoch"] += [epoch]

            print("-" * 120)
            print("Epoch {:4d}\t train_loss {:.4f},\tval_loss: {:.4f},\tbarrier parameter: {:.4f}".format(epoch, tloss, vloss, muk))
            print("-" * 120)

            if no_decrease_counter > self.patience:
                break

            check_aa = False
            if check_aa:
                aa.run_aa(self.model, self.val_loader, 0.1, 200)

            check_storage_func = False
            if check_storage_func:
                self.model.check_storage_func(self.val_loader)

        log["success"] = True
        return log, best_model
