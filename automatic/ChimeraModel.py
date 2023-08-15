import logging
import sys
import math
import pickle
import torch


class ChimeraModel:
    def __init__(self, dt, sparse_param, num_nodes, num_neuron, device, lam=1, G=1.5, Q=1, use_reload_seed=False):
        # According to paper page 7, lam controls the rate of error.
        # G is the scale parameter of omega, which can be used to control the chaotic behaviour.
        # Q is the scale parameter of eta.
        try:
            if device == "gpu":
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                    logging.error("No GPU, cuda is not available.")
                    raise Exception("There is no available GPU.")
            else:
                device = torch.device('cpu')
        except Exception as err:
            logging.error('An exception happened: ' + str(err))
            sys.exit(1)

        m = num_nodes * 2
        self.num_nodes = num_nodes
        self.num_neuron = num_neuron
        self.dt = dt
        self.z = torch.randn(num_neuron).type(torch.float64).to(device)
        self.r = torch.tanh(self.z)
        sparse_matrix = (torch.rand((num_neuron, num_neuron)) < sparse_param).to(device)
        self.omega = G * torch.mul(torch.randn((num_neuron, num_neuron)).type(torch.float64).to(device), sparse_matrix) / math.sqrt(
            num_neuron * sparse_param)
        self.eta = Q * (2 * torch.rand((num_neuron, m)) - 1).type(torch.float64).to(device)

        # d is a liner decoder, which extracts the chimera state from network.
        # P is the parameter used in RLS algorithm.
        self.d = torch.zeros((num_neuron, m)).type(torch.float64).to(device)
        self.P_inverse = torch.eye(num_neuron).type(torch.float64).to(device) / lam
        self.device = device

        # use_reload_seed is True means we use the data in init folder to initialise those random variables.
        if use_reload_seed:
            self.reload()

    # use original data from given code
    def reload(self):
        device = self.device
        self.P_inverse = torch.from_numpy(pickle.load(open("../init/P.obj", 'rb'))).to(device)
        self.eta = torch.from_numpy(pickle.load(open("../init/eta.obj", 'rb'))).to(device)
        self.omega = torch.from_numpy(pickle.load(open("../init/omega.obj", 'rb'))).to(device)
        self.z = torch.from_numpy(pickle.load(open("../init/z.obj", 'rb'))).to(device)
        self.r = torch.tanh(self.z)

    def forward(self, last_pre):
        self.z = self.z + self.dt * (-self.z + torch.matmul(self.omega, self.r) + torch.matmul(self.eta, last_pre))
        self.r = torch.tanh(self.z)
        prediction = torch.matmul(self.d.transpose(0, 1), self.r)
        return prediction

    def train(self, error):
        tmp = torch.matmul(self.P_inverse, self.r)
        self.P_inverse = self.P_inverse - (torch.outer(tmp, tmp)) / (1 + torch.dot(self.r, tmp))
        self.d = self.d - torch.outer(torch.matmul(self.P_inverse, self.r), error)
