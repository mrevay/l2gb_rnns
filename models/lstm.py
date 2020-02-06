import torch
from torch.nn import Parameter


class lstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1, batches=1, nl=None, learn_init_hidden=False, criterion=None, nBatches=1):
        super(lstm, self).__init__()

        self.nu = input_size
        self.nx = hidden_size
        self.ny = output_size
        self.layers = layers


        self.nBatches = nBatches
        # self.h0 = torch.nn.Parameter(torch.zeros(nBatches, hidden_size))

        #  nonlinearity
        if nl is None:
            self.nl = 'relu'
        else:
            self.nl = nl

        # Learnable Initial state?
        if learn_init_hidden:
            self.h0 = Parameter(torch.rand(layers, batches, self.nx))
            self.c0 = Parameter(torch.rand(layers, batches, self.nx))
        else:
            self.h0 = torch.zeros(layers, batches, self.nx)
            self.c0 = torch.zeros(layers, batches, self.nx)

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        self.recurrent_unit = torch.nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, u, h0=None, c0=None):
        # Add a new dimension corresponding to the number of hidden layers

        inputs = u.permute(0, 2, 1)
        b = u.size(0)
        if h0 is None:
            h0 = self.h0[:, 0:b, :]

        if c0 is None:
            c0 = self.c0[:, 0:b, :]
            # c0 = torch.zeros_like(h0)

        # init_state = (torch.stack((h0, h0), 0), torch.stack((c0, c0), 0))
        init_state = (h0, c0)
        # states, (hn, cn) = self.recurrent_unit(u, init_state)
        states, (hn, cn) = self.recurrent_unit(inputs, init_state)
        yest = self.output(states)

        return yest.permute(0, 2, 1)

        # return y, states

    # returns a flattened tensor of all parameters
    def flatten_params(self):
        views = []
        for p in self.parameters():
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

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

    # Used for testing a model. Data should be a dictionary containing keys
    #   "inputs" and "outputs" in order (batches x seq_len x size)
    def test(self, data, h0=None, c0=None):

        self.eval()
        with torch.no_grad():
            u = data["inputs"]
            y = data["outputs"]

            if h0 is None:
                h0 = self.h0

            if c0 is None:
                c0 = self.c0

            yest, states = self.forward(u, h0=h0, c0=c0)

            ys = y - y.mean(1).unsqueeze(2)
            error = yest - y
            NSE = error.norm() / ys.norm()
            results = {"SE": float(self.criterion(y, yest)),
                       "NSE": float(NSE),
                       "estimated": yest.detach().numpy(),
                       "inputs": u.detach().numpy(),
                       "true_outputs": y.detach().numpy(),
                       "hidden_layers": self.nx,
                       "model": "lstm"}
        return results

    def clone(self):
        copy = type(self)(self.nu, self.nx, self.ny, nBatches=self.nBatches, layers=self.layers, nl=self.nl,)
        copy.load_state_dict(self.state_dict())

        return copy

    def write_flat_params(self, theta):
        views = []
        index = 0

        for p in self.parameters():
            p.data = theta[index:index + p.numel()].view_as(p.data)
            index = index + p.numel()
