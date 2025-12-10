import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class AnfisEngine:
    def __init__(self):
        self.simulation = None

    def build_system(self):
        # --- WEJŚCIA (Znormalizowane do 0-100) ---
        
        # 1. RSI (Technika): 0-30 (Tanio), 70-100 (Drogo)
        rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
        
        # 2. VIX Rank (Strach): 0 (Spokój), 100 (Panika)
        vix = ctrl.Antecedent(np.arange(0, 101, 1), 'vix')
        
        # 3. Yield Rank (Makro): 0 (Recesja/Inwersja), 100 (Zdrowy wzrost)
        yield_curve = ctrl.Antecedent(np.arange(0, 101, 1), 'yield')

        # --- WYJŚCIE ---
        # Signal: 0 (Silne Sprzedaj / Przewartościowanie), 100 (Silne Kupuj / Okazja)
        signal = ctrl.Consequent(np.arange(0, 101, 1), 'signal')

        # --- FUNKCJE PRZYNALEŻNOŚCI ---
        
        # RSI
        rsi['oversold'] = fuzz.trimf(rsi.universe, [0, 0, 35])      # Wyprzedanie
        rsi['neutral'] = fuzz.trimf(rsi.universe, [30, 50, 70])
        rsi['overbought'] = fuzz.trimf(rsi.universe, [65, 100, 100]) # Wykupienie

        # VIX (Strach) - Tu uwaga: Wysoki VIX to często dołek cenowy (Okazja)
        vix['calm'] = fuzz.trimf(vix.universe, [0, 0, 40])
        vix['fear'] = fuzz.trimf(vix.universe, [60, 100, 100])

        # Yield Curve (Makro)
        yield_curve['inverted'] = fuzz.trimf(yield_curve.universe, [0, 0, 40]) # Ryzyko
        yield_curve['normal'] = fuzz.trimf(yield_curve.universe, [50, 100, 100]) # Zdrowo

        # Wyjście
        signal['bearish'] = fuzz.trimf(signal.universe, [0, 0, 50])
        signal['neutral'] = fuzz.trimf(signal.universe, [40, 50, 60])
        signal['bullish'] = fuzz.trimf(signal.universe, [50, 100, 100])

        # --- BAZA REGUŁ (Mózg systemu) ---
        
        rules = []
        
        # R1: "Kupuj gdy krew się leje" (Niskie RSI + Duży Strach na VIX)
        rules.append(ctrl.Rule(rsi['oversold'] & vix['fear'], signal['bullish']))
        
        # R2: "Sprzedawaj w euforii" (Wysokie RSI + VIX bardzo niski/spokojny)
        rules.append(ctrl.Rule(rsi['overbought'] & vix['calm'], signal['bearish']))
        
        # R3: "Ostrzeżenie Makro" (Inwersja krzywej = Bądź ostrożny nawet jak jest tanio)
        # To reguła, która "psuje" zabawę bykom, jeśli gospodarka ma się zawalić
        rules.append(ctrl.Rule(yield_curve['inverted'] & rsi['neutral'], signal['bearish']))

        # R4: Trend wzrostowy (Zdrowa krzywa + RSI neutralne = Powolny wzrost)
        rules.append(ctrl.Rule(yield_curve['normal'] & rsi['neutral'], signal['bullish']))

        # Budowa systemu
        system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(system)

    def compute(self, rsi_val, vix_rank, yield_rank):
        if self.simulation is None:
            self.build_system()
            
        self.simulation.input['rsi'] = np.clip(rsi_val, 0, 100)
        self.simulation.input['vix'] = np.clip(vix_rank, 0, 100)
        self.simulation.input['yield'] = np.clip(yield_rank, 0, 100)
        
        try:
            self.simulation.compute()
            return self.simulation.output['signal']
        except:
            return 50.0