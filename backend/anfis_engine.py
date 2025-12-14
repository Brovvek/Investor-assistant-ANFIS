import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class AnfisEngine:
    def __init__(self):
        self.simulations = {}
        self.weights = {}
        self.active_inputs = set()

    def _build_single_expert(self, var_name, logic_type):
        """
        logic_type: 
        'pro_trend' (High input = Buy signal) -> MACD, Yield, M2
        'counter_trend' (Low input = Buy signal) -> RSI
        'fear' (High input = Buy signal) -> VIX (Panika to okazja)
        """
        ant = ctrl.Antecedent(np.arange(0, 101, 1), var_name)
        sent = ctrl.Consequent(np.arange(0, 101, 1), 'sentiment')

        # Gauss (Dzwony)
        sigma = 20
        ant['low'] = fuzz.gaussmf(ant.universe, 20, sigma)
        ant['medium'] = fuzz.gaussmf(ant.universe, 50, sigma)
        ant['high'] = fuzz.gaussmf(ant.universe, 80, sigma)

        sigma_out = 15
        sent['sell'] = fuzz.gaussmf(sent.universe, 20, sigma_out)
        sent['neutral'] = fuzz.gaussmf(sent.universe, 50, sigma_out)
        sent['buy'] = fuzz.gaussmf(sent.universe, 80, sigma_out)

        rules = []
        if logic_type == 'pro_trend': 
            rules.append(ctrl.Rule(ant['high'], sent['buy']))
            rules.append(ctrl.Rule(ant['medium'], sent['neutral']))
            rules.append(ctrl.Rule(ant['low'], sent['sell']))
        elif logic_type == 'counter_trend': 
            rules.append(ctrl.Rule(ant['low'], sent['buy']))
            rules.append(ctrl.Rule(ant['medium'], sent['neutral']))
            rules.append(ctrl.Rule(ant['high'], sent['sell']))
        elif logic_type == 'fear':
            rules.append(ctrl.Rule(ant['high'], sent['buy']))
            rules.append(ctrl.Rule(ant['medium'], sent['neutral']))
            rules.append(ctrl.Rule(ant['low'], sent['sell']))

        system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(system)

    def build_system(self, user_config):
        self.active_inputs = set()
        self.weights = {}
        self.simulations = {}

        # Mapa logiki dla wskaźników
        indicators_map = {
            'rsi': 'counter_trend',
            'vix': 'fear',
            'yield': 'pro_trend',
            'macd': 'pro_trend',
            'm2': 'pro_trend'
        }

        for key, logic in indicators_map.items():
            cfg = user_config.get(key, {'enabled': False, 'weight': 0.0})
            
            if cfg.get('enabled', False):
                self.active_inputs.add(key)
                # Zabezpieczenie przed wagą 0
                w = float(cfg.get('weight', 0.0))
                self.weights[key] = w if w > 0 else 0.001 
                self.simulations[key] = self._build_single_expert(key, logic)

    def compute(self, inputs_dict):
        total_score = 0.0
        total_weight = 0.0

        for key, sim in self.simulations.items():
            if key in inputs_dict:
                val = np.clip(inputs_dict[key], 0, 100)
                sim.input[key] = val
                try:
                    sim.compute()
                    w = self.weights[key]
                    total_score += sim.output['sentiment'] * w
                    total_weight += w
                except: pass

        if total_weight == 0: return 50.0
        return total_score / total_weight