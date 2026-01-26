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
        """
        Fit coefficients using weighted least squares for each rule.
        weights: (batch, d_rule)
        x: (batch, d_in)
        y_actual: (batch,) or (batch, d_out)
        coeff shape should be: (d_rule, d_out, d_in+1)
        """
        batch_size = x.shape[0]
        n_rules = weights.shape[1]
        n_in = x.shape[1]
        
        # Ensure y_actual is 2D: (batch, d_out)
        if y_actual.dim() == 1:
            y_actual = y_actual.unsqueeze(1)
        d_out = y_actual.shape[1]
        
        # Check for NaN/Inf in inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            print('Warning: Input x contains NaN/Inf values - cleaning...')
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
        if torch.isnan(y_actual).any() or torch.isinf(y_actual).any():
            print('Warning: Output y_actual contains NaN/Inf values - cleaning...')
            y_actual = torch.where(torch.isnan(y_actual), torch.zeros_like(y_actual), y_actual)
            y_actual = torch.where(torch.isinf(y_actual), torch.zeros_like(y_actual), y_actual)
        
        # Clean NaN/Inf from weights
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print('Warning: Weights contain NaN/Inf values - cleaning...')
            weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
            weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
            # Renormalize weights after cleaning
            row_sums = weights.sum(dim=1, keepdim=True)
            zero_mask = row_sums == 0
            if zero_mask.any():
                weights[zero_mask.squeeze(1)] = torch.ones_like(weights[zero_mask.squeeze(1)]) / weights.shape[1]
            weights = F.normalize(weights, p=1, dim=1)
        
        # Add bias term to inputs
        x_plus = torch.cat([x, torch.ones(batch_size, 1, dtype=x.dtype, device=x.device)], dim=1)  # (batch, n_in+1)
        
        # Initialize coefficient matrix for all rules
        coeff_all = torch.zeros(n_rules, d_out, n_in + 1, dtype=x.dtype, device=x.device)
        
        try:
            # Solve weighted least squares for each rule separately
            for rule_idx in range(n_rules):
                # Get weights for this rule across all samples
                w = weights[:, rule_idx]  # (batch,)
                
                # Filter out very small weights to avoid numerical issues
                # Only use samples where weight > threshold
                weight_threshold = 1e-8
                valid_mask = w > weight_threshold
                n_valid = valid_mask.sum().item()
                
                # Need at least n_in+1 samples to solve the system
                if n_valid < n_in + 1:
                    # Not enough valid samples, skip this rule
                    continue
                
                # Get valid samples
                w_valid = w[valid_mask]
                x_valid = x_plus[valid_mask]  # (n_valid, n_in+1)
                y_valid = y_actual[valid_mask]  # (n_valid, d_out)
                
                # Normalize weights to [0, 1]
                w_norm = w_valid / (w_valid.max() + 1e-12)
                
                # Use sqrt of weights for numerical stability
                sqrt_w = torch.sqrt(w_norm)  # (n_valid,)
                
                # Scale inputs and outputs by sqrt(weights)
                weighted_x = x_valid * sqrt_w.unsqueeze(1)  # (n_valid, n_in+1)
                weighted_y = y_valid * sqrt_w.unsqueeze(1)  # (n_valid, d_out)
                
                # Check for NaN/Inf after weighting
                if torch.isnan(weighted_x).any() or torch.isinf(weighted_x).any():
                    print(f'Rule {rule_idx}: weighted_x contains NaN/Inf')
                    continue
                if torch.isnan(weighted_y).any() or torch.isinf(weighted_y).any():
                    print(f'Rule {rule_idx}: weighted_y contains NaN/Inf')
                    continue
                
                try:
                    # Solve using normal equations for better numerical stability
                    # X^T X coeff = X^T y
                    XtX = weighted_x.t() @ weighted_x  # (n_in+1, n_in+1)
                    Xty = weighted_x.t() @ weighted_y  # (n_in+1, d_out)
                    
                    # Add ridge regularization for stability
                    ridge_lambda = 1e-6 * torch.trace(XtX) / (n_in + 1)
                    XtX_ridge = XtX + ridge_lambda * torch.eye(n_in + 1, dtype=XtX.dtype, device=XtX.device)
                    
                    # Solve normal equations
                    coeff_rule = torch.linalg.solve(XtX_ridge, Xty)  # (n_in+1, d_out)
                    coeff_all[rule_idx] = coeff_rule.t()  # Transpose to (d_out, n_in+1)
                    
                except (RuntimeError, torch.linalg.LinAlgError) as solve_err:
                    # Normal equations failed, try direct lstsq
                    try:
                        solution = torch.linalg.lstsq(weighted_x.to(torch.float64), weighted_y.to(torch.float64), rcond=1e-6).solution
                        coeff_all[rule_idx] = solution.t().to(weighted_x.dtype)
                    except RuntimeError as lstsq_err:
                        # lstsq also failed, try pseudoinverse with higher tolerance
                        try:
                            # Use float64 for better numerical precision
                            pinv_matrix = torch.linalg.pinv(weighted_x.to(torch.float64), rcond=1e-5)
                            sol = (pinv_matrix @ weighted_y.to(torch.float64)).to(weighted_x.dtype)
                            coeff_all[rule_idx] = sol.t()
                        except Exception as pinv_err:
                            # All methods failed, keep zero coefficients
                            print(f'Rule {rule_idx}: All solving methods failed')
                            pass
                    
        except Exception as e:
            print('Internal error in fit_coeff:', e)
            import traceback
            traceback.print_exc()
            return
        
        # Set coefficient - shape should be (d_rule, d_out, d_in+1)
        self.coeff = coeff_all

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
        
        # Clean NaN/Inf values from raw weights before normalization
        self.raw_weights = torch.where(torch.isnan(self.raw_weights), torch.zeros_like(self.raw_weights), self.raw_weights)
        self.raw_weights = torch.where(torch.isinf(self.raw_weights), torch.zeros_like(self.raw_weights), self.raw_weights)
        
        # Replace any all-zero rows (all rules failed) with uniform distribution
        row_sums = self.raw_weights.sum(dim=1, keepdim=True)
        zero_mask = row_sums == 0
        if zero_mask.any():
            self.raw_weights[zero_mask.squeeze(1)] = torch.ones_like(self.raw_weights[zero_mask.squeeze(1)]) / self.raw_weights.shape[1]
        
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
