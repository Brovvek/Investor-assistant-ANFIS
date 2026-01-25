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
        Buduje eksperta uÅ¼ywajÄ…c Å‚agodnych funkcji GAUSSA (Krzywe dzwonowe).
        Zapewnia to pÅ‚ynniejsze przejÅ›cia i mniejszÄ… wraÅ¼liwoÅ›Ä‡ na szum.
        """
        ant = ctrl.Antecedent(np.arange(0, 101, 1), var_name)
        sent = ctrl.Consequent(np.arange(0, 101, 1), 'sentiment')

        sigma_in = 10
        
        # Funkcje przynaleÅ¼noÅ›ci (Input)
        ant['low'] = fuzz.gaussmf(ant.universe, 20, sigma_in)
        ant['medium'] = fuzz.gaussmf(ant.universe, 50, sigma_in)
        ant['high'] = fuzz.gaussmf(ant.universe, 80, sigma_in)

        # Funkcje przynaleÅ¼noÅ›ci (Output - Sentyment)
        sigma_out = 15
        sent['sell'] = fuzz.gaussmf(sent.universe, 20, sigma_out)
        sent['neutral'] = fuzz.gaussmf(sent.universe, 50, sigma_out)
        sent['buy'] = fuzz.gaussmf(sent.universe, 80, sigma_out)
        
        # Metoda defuzzyfikacji (Centroid jest standardem dla Gaussa)
        sent.defuzzify_method = 'centroid'

        rules = []
        if logic_type == 'pro_trend': 
            # High input = Buy (np. MACD roÅ›nie)
            rules.append(ctrl.Rule(ant['high'], sent['buy']))
            rules.append(ctrl.Rule(ant['medium'], sent['neutral']))
            rules.append(ctrl.Rule(ant['low'], sent['sell']))
        else: 
            # Low input = Buy (np. RSI nisko / Contra)
            rules.append(ctrl.Rule(ant['low'], sent['buy']))
            rules.append(ctrl.Rule(ant['medium'], sent['neutral']))
            rules.append(ctrl.Rule(ant['high'], sent['sell']))

        system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(system)

    def build_system(self, user_config):
        self.active_inputs = set()
        self.weights = {}
        self.simulations = {}

        for feature_name, settings in user_config.items():
            if not settings.get('enabled', False):
                continue
            
            weight = float(settings.get('weight', 1.0))
            if weight <= 0: continue

            # ObsÅ‚uga dynamicznych kierunkÃ³w (z Fazy 3)
            direction = settings.get('direction', 1)
            logic_type = 'pro_trend' if direction > 0 else 'counter_trend'

            self.active_inputs.add(feature_name)
            self.weights[feature_name] = weight
            self.simulations[feature_name] = self._build_single_expert(feature_name, logic_type)

    def compute(self, inputs_dict):
        total_score = 0.0
        total_weight = 0.0
        
        input_processed = False

        for key, sim in self.simulations.items():
            if key in inputs_dict:
                val = np.clip(inputs_dict[key], 0, 100)
                sim.input[key] = val
                try:
                    sim.compute()
                    output = sim.output['sentiment']
                    
                    w = self.weights[key]
                    total_score += output * w
                    total_weight += w
                    input_processed = True
                except: 
                    # Fallback
                    pass

        if not input_processed or total_weight == 0: 
            return 50.0
        
        # Czysta Å›rednia waÅ¼ona (Bez sztucznego rozciÄ…gania SigmoidÄ…)
        final_result = total_score / total_weight
        
        return final_result