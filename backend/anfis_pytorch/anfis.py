import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import itertools

class FuzzifyVariable(nn.Module):
    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        return len(self.mfdefs)

    def members(self):
        return self.mfdefs.values()

    def forward(self, x):
        y_pred = torch.cat([mf(x) for mf in self.members()], dim=1)
        return y_pred

class FuzzifyLayer(nn.Module):
    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        self.varmfs = nn.ModuleDict(OrderedDict(zip(self.varnames, varmfs)))

    def forward(self, x):
        outputs = []
        for i, (name, var) in enumerate(self.varmfs.items()):
            outputs.append(var(x[:, i:i + 1]))
        return outputs

class AntecedentLayer(nn.Module):
    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        mf_indices = [range(var.num_mfs) for var in varlist]
        self.mf_indices = list(itertools.product(*mf_indices))

    def num_rules(self):
        return len(self.mf_indices)

    def forward(self, x):
        batch_indices = []
        for idx in self.mf_indices:
            batch_indices.append(torch.stack([x[i][:, j] for i, j in enumerate(idx)], dim=1))
        
        # Obliczanie siły reguły (iloczyn)
        rules = torch.stack([torch.prod(b, dim=1) for b in batch_indices], dim=1)
        return rules

class ConsequentLayer(nn.Module):
    def __init__(self, d_in, d_rule, d_out):
        super(ConsequentLayer, self).__init__()
        self._d_in = d_in
        self._d_rule = d_rule
        self._d_out = d_out
        self.coeffs = nn.Parameter(torch.zeros(d_rule, d_in + 1, d_out))

    def forward(self, x):
        batch_size = x.shape[0]
        x_plus_bias = torch.cat([x, torch.ones(batch_size, 1)], dim=1)
        
        # Przygotowanie wymiarów pod mnożenie (Batch, Rules, Inputs, 1)
        # To naprawia błąd "Broadcasting" przy dużej liczbie reguł
        x_exp = x_plus_bias.unsqueeze(1).unsqueeze(-1).expand(batch_size, self._d_rule, self._d_in + 1, 1)
        coeffs_exp = self.coeffs.unsqueeze(0).expand(batch_size, self._d_rule, self._d_in + 1, self._d_out)
        
        # Wynik: (Batch, Rules, 1)
        y_pred = torch.sum(x_exp * coeffs_exp, dim=2)
        return y_pred

class AnfisNet(nn.Module):
    def __init__(self, description, invardefs, outvars=['y']):
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvars = outvars
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.mfdefs = mfdefs

        fuzzify_layer = FuzzifyLayer(mfdefs, varnames)
        rules_layer = AntecedentLayer(mfdefs)
        consequent_layer = ConsequentLayer(self.num_in, rules_layer.num_rules(), len(outvars))

        self.layer = nn.ModuleDict(OrderedDict([
            ('fuzzify', fuzzify_layer),
            ('rules', rules_layer),
            ('consequent', consequent_layer)
        ]))

    def forward(self, x):
        # 1. Rozmywanie
        fuzzified = self.layer['fuzzify'](x)
        
        # 2. Obliczenie wag reguł (Batch, Rules)
        w = self.layer['rules'](fuzzified)
        
        # 3. Normalizacja wag
        w_norm = F.normalize(w, p=1, dim=1)
        
        # 4. Obliczenie wartości reguł TSK (Batch, Rules, 1)
        y_tsk = self.layer['consequent'](x)
        
        # 5. Defuzzify (Średnia ważona)
        # Mnożenie macierzy: (Batch, 1, Rules) * (Batch, Rules, 1) -> (Batch, 1, 1)
        y_pred = torch.bmm(w_norm.unsqueeze(1), y_tsk)
        
        return y_pred.squeeze(1).squeeze(1)