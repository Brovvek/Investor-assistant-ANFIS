#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: the ANFIS layers
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
    Modified for Investor Assistant ANFIS Pro
'''

import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


dtype = torch.float


class FuzzifyVariable(torch.nn.Module):
    '''
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    '''
    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        return len(self.mfdefs)

    def members(self):
        return self.mfdefs.items()

    def pad_to(self, new_size):
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield(mfname, yvals)

    def forward(self, x):
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding).to(x.device)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        A list of fuzzy variables, representing the inputs to the FIS.
    '''
    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        return len(self.varmfs)

    @property
    def max_mfs(self):
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                         mfdef.__class__.__name__,
                         ', '.join(['{}={:.4f}'.format(n, p.item())
                                   for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)

    def forward(self, x):
        assert x.shape[1] == self.num_in,\
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i+1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class AntecedentLayer(torch.nn.Module):
    '''
        Form the 'rules' by taking all possible combinations of the MFs.
    '''
    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        mf_count = [var.num_mfs for var in varlist]
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))

    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        batch_indices = batch_indices.to(x.device)
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        rules = torch.prod(ants, dim=2)
        return rules


class ConsequentLayer(torch.nn.Module):
    '''
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
    '''
    def __init__(self, d_in, d_rule, d_out):
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in+1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}'\
            .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1).to(x.device)], dim=1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        weighted_x[weighted_x == 0] = 1e-12
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        try:
            coeff_2d = torch.linalg.lstsq(weighted_x_2d, y_actual_2d).solution
        except Exception as e:
            print('Internal error in lstsq', e)
            return
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1]+1, -1)\
            .transpose(1, 2)

    def forward(self, x):
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1).to(x.device)], dim=1)
        y_pred = torch.matmul(self.coeff.to(x.device), x_plus.t())
        return y_pred.transpose(0, 2)


class PlainConsequentLayer(ConsequentLayer):
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''
    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        pass  # Using BP to learn coefficients


class AnfisNet(torch.nn.Module):
    '''
        This is a container for the 5 layers of the ANFIS net.
    '''
    def __init__(self, description, invardefs, outvarnames, device, hybrid=True):
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.device = device
        self.hybrid = hybrid
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs)),
            ('consequent', cl),
            ]))

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        return self.outvarnames

    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' '*9+'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)

    def forward(self, x):
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


# ========== MEMBERSHIP FUNCTIONS ==========

def _mk_param(val):
    '''Make a torch parameter from a scalar value'''
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMembFunc(torch.nn.Module):
    '''
        Gaussian membership functions, defined by two parameters:
            mu, the mean (center)
            sigma, the standard deviation.
    '''
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val

    def pretty(self):
        return 'GaussMembFunc mu={:.4f} sigma={:.4f}'.format(
            self.mu.item(), self.sigma.item())


class BellMembFunc(torch.nn.Module):
    '''
        Generalised Bell membership function
    '''
    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMembFunc.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def pretty(self):
        return 'BellMembFunc a={:.4f} b={:.4f} c={:.4f}'.format(
            self.a.item(), self.b.item(), self.c.item())


class TriangularMembFunc(torch.nn.Module):
    '''
        Triangular membership function
    '''
    def __init__(self, a, b, c):
        super(TriangularMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))

    def forward(self, x):
        # Safe implementation to avoid NaN
        left = (x - self.a) / (self.b - self.a + 1e-9)
        right = (self.c - x) / (self.c - self.b + 1e-9)
        return torch.clamp(torch.min(left, right), min=0, max=1)

    def pretty(self):
        return 'TriangularMembFunc a={:.4f} b={:.4f} c={:.4f}'.format(
            self.a.item(), self.b.item(), self.c.item())


def make_gauss_mfs(sigma, mu_list):
    '''Return a list of gaussian mfs, same sigma, list of means'''
    return [GaussMembFunc(mu, sigma) for mu in mu_list]


def make_bell_mfs(a, b, clist):
    '''Return a list of bell mfs, same (a,b), list of centers'''
    return [BellMembFunc(a, b, c) for c in clist]


def make_tri_mfs(width, clist):
    '''Return a list of triangular mfs, same width, list of centers'''
    return [TriangularMembFunc(c-width/2, c, c+width/2) for c in clist]


def make_anfis(x, num_mfs=5, num_out=1, hybrid=True, mf_type='gauss', var_names=None):
    '''
        Make an ANFIS model, auto-calculating the MFs.
    '''
    num_invars = x.shape[1]
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    ranges = maxvals - minvals
    
    invars = []
    for i in range(num_invars):
        if var_names and i < len(var_names):
            vname = var_names[i]
        else:
            vname = 'x{}'.format(i)
            
        if num_mfs == 1:
            mulist = [torch.linspace(minvals[i], maxvals[i], num_mfs + 1).tolist()[0]]
        else:
            mulist = torch.linspace(minvals[i], maxvals[i], num_mfs).tolist()
        
        sigma = ranges[i] / num_mfs
        
        if mf_type == 'gauss':
            mfs = make_gauss_mfs(sigma, mulist)
        elif mf_type == 'bell':
            mfs = make_bell_mfs(sigma, 2.0, mulist)
        elif mf_type == 'tri':
            width = ranges[i] / (num_mfs - 1) if num_mfs > 1 else ranges[i]
            mfs = make_tri_mfs(width, mulist)
        else:
            mfs = make_gauss_mfs(sigma, mulist)
            
        invars.append((vname, mfs))
    
    outvars = ['y{}'.format(i) for i in range(num_out)]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AnfisNet('ANFIS Price Predictor', invars, outvars, hybrid=hybrid, device=device)
    return model
