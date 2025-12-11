import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class AnfisEngine:
    def __init__(self):
        self.simulation = None
        self.active_inputs = set() # Zbiór do pamiętania, co jest włączone

    def build_system(self, user_config):
        """
        Buduje system dynamicznie, USUWAJĄC wyłączone wskaźniki z logiki reguł.
        """
        self.active_inputs = set() # Resetujemy listę aktywnych wejść
        
        # 1. Pobieramy wagi i status (enabled)
        def get_cfg(key):
            cfg = user_config.get(key, {'enabled': True, 'weight': 1.0})
            val = cfg['weight'] if cfg['enabled'] else 0.0
            # Jeśli waga > 0, dodajemy do listy aktywnych
            if val > 0:
                self.active_inputs.add(key)
            return val

        w_rsi = get_cfg('rsi')
        w_vix = get_cfg('vix')
        w_yield = get_cfg('yield')

        # Jeśli wszystko wyłączone -> brak systemu
        if not self.active_inputs:
            self.simulation = None
            return

        # 2. Definicja zmiennych
        rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
        vix = ctrl.Antecedent(np.arange(0, 101, 1), 'vix')
        yield_curve = ctrl.Antecedent(np.arange(0, 101, 1), 'yield')
        sentiment = ctrl.Consequent(np.arange(0, 101, 1), 'sentiment')

        # Funkcje przynależności (Gauss)
        sigma = 20
        for var in [rsi, vix, yield_curve]:
            var['low'] = fuzz.gaussmf(var.universe, 20, sigma)
            var['medium'] = fuzz.gaussmf(var.universe, 50, sigma)
            var['high'] = fuzz.gaussmf(var.universe, 80, sigma)

        sigma_out = 15
        sentiment['strong_sell'] = fuzz.gaussmf(sentiment.universe, 10, sigma_out)
        sentiment['sell'] = fuzz.gaussmf(sentiment.universe, 30, sigma_out)
        sentiment['neutral'] = fuzz.gaussmf(sentiment.universe, 50, sigma_out)
        sentiment['buy'] = fuzz.gaussmf(sentiment.universe, 70, sigma_out)
        sentiment['strong_buy'] = fuzz.gaussmf(sentiment.universe, 90, sigma_out)

        # 3. Dynamiczne tworzenie reguł
        rules = []

        def create_dynamic_rule(conditions, consequent):
            # Filtrujemy tylko aktywne wskaźniki (waga > 0)
            active_terms = [term for term, w in conditions if w > 0]
            active_weights = [w for term, w in conditions if w > 0]

            if not active_terms:
                return None 

            antecedent = active_terms[0]
            for term in active_terms[1:]:
                antecedent = antecedent & term
            
            avg_weight = sum(active_weights) / len(active_weights)
            
            rule = ctrl.Rule(antecedent, consequent)
            rule.weight = avg_weight
            return rule

        # --- REGUŁY ---

        # R1: Super Okazja (RSI Low + VIX High + Yield High)
        r1 = create_dynamic_rule([
            (rsi['low'], w_rsi), 
            (vix['high'], w_vix), 
            (yield_curve['high'], w_yield)
        ], sentiment['strong_buy'])
        if r1: rules.append(r1)

        # R2: Krach / Recesja (RSI Low + Yield Low)
        r2 = create_dynamic_rule([
            (rsi['low'], w_rsi), 
            (yield_curve['low'], w_yield)
        ], sentiment['buy'])
        if r2: rules.append(r2)

        # R3: Bańka (RSI High + VIX Low)
        r3 = create_dynamic_rule([
            (rsi['high'], w_rsi), 
            (vix['low'], w_vix)
        ], sentiment['strong_sell'])
        if r3: rules.append(r3)

        # R4: Nuda (RSI Medium + VIX Medium)
        r4 = create_dynamic_rule([
            (rsi['medium'], w_rsi), 
            (vix['medium'], w_vix)
        ], sentiment['neutral'])
        if r4: rules.append(r4)

        # R5: Ostrzeżenie Makro (Yield Low + RSI Medium)
        r5 = create_dynamic_rule([
            (yield_curve['low'], w_yield),
            (rsi['medium'], w_rsi)
        ], sentiment['sell'])
        if r5: rules.append(r5)

        # R6 (Awaryjna dla VIX): Jeśli tylko VIX jest włączony
        if w_vix > 0 and w_rsi == 0 and w_yield == 0:
            rules.append(ctrl.Rule(vix['high'], sentiment['buy']))
            rules.append(ctrl.Rule(vix['low'], sentiment['sell']))
            
        # R7 (Awaryjna dla RSI): Jeśli tylko RSI jest włączone
        if w_rsi > 0 and w_vix == 0 and w_yield == 0:
            rules.append(ctrl.Rule(rsi['low'], sentiment['buy']))
            rules.append(ctrl.Rule(rsi['high'], sentiment['sell']))

        # Budowa systemu
        if rules:
            system = ctrl.ControlSystem(rules)
            self.simulation = ctrl.ControlSystemSimulation(system)
        else:
            self.simulation = None

    def compute(self, rsi_val, vix_rank, yield_rank):
        if self.simulation is None:
            return 50.0
            
        # --- POPRAWKA: Podajemy tylko te dane, które są aktywne ---
        
        if 'rsi' in self.active_inputs:
            self.simulation.input['rsi'] = np.clip(rsi_val, 0, 100)
            
        if 'vix' in self.active_inputs:
            self.simulation.input['vix'] = np.clip(vix_rank, 0, 100)
            
        if 'yield' in self.active_inputs:
            self.simulation.input['yield'] = np.clip(yield_rank, 0, 100)
        
        try:
            self.simulation.compute()
            return self.simulation.output['sentiment']
        except ValueError:
            # Dodatkowe zabezpieczenie, gdyby skfuzzy nadal protestowało
            return 50.0
        except Exception:
            return 50.0