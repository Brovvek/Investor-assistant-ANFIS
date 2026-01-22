import torch
import torch.nn as nn
from .anfis import AnfisNet

def make_param(val):
    if isinstance(val, torch.Tensor):
        val = val.clone().detach()
    else:
        val = torch.tensor(val, dtype=torch.float)
    
    return nn.Parameter(val)

class GaussMembFunc(nn.Module):
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', make_param(mu))
        self.register_parameter('sigma', make_param(sigma))

    def forward(self, x):
        return torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))

def make_anfis(x, num_mfs=3, num_out=1):
    num_invars = x.shape[1]
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    ranges = maxvals - minvals
    
    invars = []
    for i in range(num_invars):
        sigma = ranges[i] / num_mfs
        mu_list = torch.linspace(minvals[i], maxvals[i], num_mfs).tolist()
        mfs = [GaussMembFunc(mu, sigma) for mu in mu_list]
        invars.append(('x{}'.format(i), mfs))
    
    model = AnfisNet('Simple ANFIS', invars, ['y{}'.format(i) for i in range(num_out)])
    return model