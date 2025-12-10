import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class AnfisEngine:
    def __init__(self):
        self.simulation = None

    def build_system(self):
        # --- Zmienne wejściowe ---
        # Inflacja (CPI YoY) - zakres 0% do 15%
        inflation = ctrl.Antecedent(np.arange(0, 15, 0.1), 'inflation')
        # Bezrobocie (UNRATE) - zakres 0% do 15%
        unemployment = ctrl.Antecedent(np.arange(0, 15, 0.1), 'unemployment')
        
        # --- Zmienna wyjściowa (Twój Oscylator) ---
        sentiment = ctrl.Consequent(np.arange(0, 101, 1), 'sentiment')

        # --- Funkcje przynależności (Automatyczne) ---
        inflation.automf(3, names=['low', 'medium', 'high'])
        unemployment.automf(3, names=['low', 'medium', 'high'])

        # Definicja wyjścia
        sentiment['bearish'] = fuzz.trimf(sentiment.universe, [0, 0, 50])
        sentiment['neutral'] = fuzz.trimf(sentiment.universe, [25, 50, 75])
        sentiment['bullish'] = fuzz.trimf(sentiment.universe, [50, 100, 100])

        # --- Reguły Logiczne (Przykładowa logika rynkowa) ---
        # 1. Niska inflacja + Niskie bezrobocie = Idealne warunki (Hossa)
        rule1 = ctrl.Rule(inflation['low'] & unemployment['low'], sentiment['bullish'])
        # 2. Wysoka inflacja + Wysokie bezrobocie = Stagflacja (Bessa)
        rule2 = ctrl.Rule(inflation['high'] & unemployment['high'], sentiment['bearish'])
        # 3. Wysoka inflacja (niezależnie od bezrobocia) = Strach przed stopami proc.
        rule3 = ctrl.Rule(inflation['high'], sentiment['bearish'])
        # 4. Inne przypadki = Neutralne
        rule4 = ctrl.Rule(inflation['medium'], sentiment['neutral'])

        sentiment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.simulation = ctrl.ControlSystemSimulation(sentiment_ctrl)

    def compute_sentiment(self, inflation_val, unemployment_val):
        if self.simulation is None:
            self.build_system()
            
        # Zabezpieczenie przed wartościami spoza zakresu (clip)
        i_val = np.clip(inflation_val, 0, 14.9)
        u_val = np.clip(unemployment_val, 0, 14.9)
        
        self.simulation.input['inflation'] = i_val
        self.simulation.input['unemployment'] = u_val
        
        try:
            self.simulation.compute()
            return self.simulation.output['sentiment']
        except:
            return 50.0 # Wartość neutralna w razie błędu