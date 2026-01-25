"""
Fuzzy Expert System (FIS) - Mamdani Type
========================================

WAŻNE: To NIE jest ANFIS!

Ten moduł implementuje KLASYCZNY SYSTEM ROZMYTY (Fuzzy Inference System)
typu Mamdani z eksperckimi regułami. NIE MA tutaj uczenia maszynowego.

Terminologia:
- FIS (Fuzzy Inference System) - system wnioskowania rozmytego
- FLC (Fuzzy Logic Controller) - kontroler logiki rozmytej
- Mamdani - typ systemu z rozmytym wyjściem (w przeciwieństwie do Sugeno/TSK)

Co ten moduł robi:
- Zamienia wartości wskaźników (0-100) na "sentyment rynkowy" (0-100)
- Używa funkcji przynależności Gaussa (LOW, MEDIUM, HIGH)
- Stosuje reguły IF-THEN zdefiniowane przez eksperta (nie uczone!)
- Agreguje wyniki z wielu wskaźników średnią ważoną

Czym to NIE jest:
- To NIE jest ANFIS (brak uczenia, brak adaptacji)
- To NIE jest sieć neuronowa
- Parametry NIE są optymalizowane przez gradient descent

Prawdziwy ANFIS znajduje się w: anfis_torch.py i anfis_ml_engine_v2.py
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyExpertSystem:
    """
    Klasyczny System Ekspertowy oparty na Logice Rozmytej (FIS Mamdani).
    
    Generuje "oscylator sentymentu" (0-100) na podstawie wskaźników technicznych.
    Używa sztywnych reguł zdefiniowanych przez eksperta - BEZ UCZENIA.
    
    Attributes:
        simulations: Słownik symulacji dla każdego wskaźnika
        weights: Wagi poszczególnych wskaźników
        active_inputs: Zbiór aktywnych wskaźników
    
    Example:
        >>> fis = FuzzyExpertSystem()
        >>> config = {
        ...     'RSI': {'enabled': True, 'weight': 1.0, 'direction': -1},
        ...     'MACD': {'enabled': True, 'weight': 1.5, 'direction': 1}
        ... }
        >>> fis.build_system(config)
        >>> sentiment = fis.compute({'RSI': 25, 'MACD': 70})
        >>> print(f"Sentyment: {sentiment:.1f}")  # np. 76.3
    """
    
    def __init__(self):
        self.simulations = {}
        self.weights = {}
        self.active_inputs = set()

    def _build_single_expert(self, var_name, logic_type):
        """
        Buduje pojedynczy ekspert rozmyty dla jednego wskaźnika.
        
        Używa funkcji przynależności Gaussa (krzywe dzwonowe) dla płynnych
        przejść między kategoriami LOW, MEDIUM, HIGH.
        
        Args:
            var_name: Nazwa zmiennej wejściowej (np. 'RSI')
            logic_type: 'pro_trend' lub 'counter_trend'
                - pro_trend: HIGH input → BUY (np. MACD rośnie = dobrze)
                - counter_trend: LOW input → BUY (np. RSI niski = oversold = kupuj)
        
        Returns:
            ControlSystemSimulation: Gotowa symulacja systemu rozmytego
        """
        # Definicja zmiennej wejściowej (Antecedent) - zakres 0-100
        ant = ctrl.Antecedent(np.arange(0, 101, 1), var_name)
        
        # Definicja zmiennej wyjściowej (Consequent) - sentyment 0-100
        sent = ctrl.Consequent(np.arange(0, 101, 1), 'sentiment')

        # Parametry funkcji Gaussa dla wejścia
        sigma_in = 10  # Szerokość krzywej
        
        # Funkcje przynależności wejścia (Input Membership Functions)
        # Centra: LOW=20, MEDIUM=50, HIGH=80
        ant['low'] = fuzz.gaussmf(ant.universe, 20, sigma_in)
        ant['medium'] = fuzz.gaussmf(ant.universe, 50, sigma_in)
        ant['high'] = fuzz.gaussmf(ant.universe, 80, sigma_in)

        # Funkcje przynależności wyjścia (Output Membership Functions)
        sigma_out = 15  # Szerokość krzywej wyjściowej
        sent['sell'] = fuzz.gaussmf(sent.universe, 20, sigma_out)
        sent['neutral'] = fuzz.gaussmf(sent.universe, 50, sigma_out)
        sent['buy'] = fuzz.gaussmf(sent.universe, 80, sigma_out)
        
        # Metoda defuzzyfikacji - centroid (środek ciężkości)
        sent.defuzzify_method = 'centroid'

        # Definicja reguł IF-THEN
        rules = []
        if logic_type == 'pro_trend': 
            # Logika pro-trendowa: wysoki input = sygnał kupna
            # Przykłady: MACD (rosnący = momentum wzrostowe), M2 (płynność rośnie)
            rules.append(ctrl.Rule(ant['high'], sent['buy']))
            rules.append(ctrl.Rule(ant['medium'], sent['neutral']))
            rules.append(ctrl.Rule(ant['low'], sent['sell']))
        else: 
            # Logika kontrariańska: niski input = sygnał kupna
            # Przykłady: RSI (niski = wyprzedany), VIX (wysoki strach = okazja)
            rules.append(ctrl.Rule(ant['low'], sent['buy']))
            rules.append(ctrl.Rule(ant['medium'], sent['neutral']))
            rules.append(ctrl.Rule(ant['high'], sent['sell']))

        # Budowa systemu sterowania
        system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(system)

    def build_system(self, user_config):
        """
        Buduje kompletny system ekspertowy na podstawie konfiguracji użytkownika.
        
        Args:
            user_config: Słownik konfiguracji wskaźników, np.:
                {
                    'RSI': {'enabled': True, 'weight': 1.0, 'direction': -1},
                    'MACD': {'enabled': True, 'weight': 1.5, 'direction': 1},
                    'VIX': {'enabled': False, 'weight': 1.0, 'direction': -1}
                }
                
                - enabled: czy wskaźnik jest aktywny
                - weight: waga wskaźnika (0.0 - 3.0)
                - direction: +1 = pro_trend, -1 = counter_trend
        """
        # Reset stanu
        self.active_inputs = set()
        self.weights = {}
        self.simulations = {}

        for feature_name, settings in user_config.items():
            # Pomiń wyłączone wskaźniki
            if not settings.get('enabled', False):
                continue
            
            # Pomiń wskaźniki z zerową wagą
            weight = float(settings.get('weight', 1.0))
            if weight <= 0:
                continue

            # Określ typ logiki na podstawie kierunku
            direction = settings.get('direction', 1)
            logic_type = 'pro_trend' if direction > 0 else 'counter_trend'

            # Zbuduj eksperta dla tego wskaźnika
            self.active_inputs.add(feature_name)
            self.weights[feature_name] = weight
            self.simulations[feature_name] = self._build_single_expert(feature_name, logic_type)

    def compute(self, inputs_dict):
        """
        Oblicza końcowy sentyment jako średnią ważoną z wszystkich ekspertów.
        
        Args:
            inputs_dict: Słownik wartości wskaźników, np.:
                {'RSI': 25, 'MACD': 70, 'VIX': 18}
                Wartości powinny być w zakresie 0-100.
        
        Returns:
            float: Sentyment rynkowy (0-100)
                - 0-30: Strefa wyprzedania (sygnał kupna)
                - 30-70: Strefa neutralna
                - 70-100: Strefa wykupienia (sygnał sprzedaży)
        """
        total_score = 0.0
        total_weight = 0.0
        input_processed = False

        for key, sim in self.simulations.items():
            if key in inputs_dict:
                # Ogranicz wartość do zakresu 0-100
                val = np.clip(inputs_dict[key], 0, 100)
                sim.input[key] = val
                
                try:
                    # Wykonaj wnioskowanie rozmyte
                    sim.compute()
                    output = sim.output['sentiment']
                    
                    # Dodaj do średniej ważonej
                    w = self.weights[key]
                    total_score += output * w
                    total_weight += w
                    input_processed = True
                except Exception:
                    # Fallback w przypadku błędu (np. puste reguły)
                    pass

        # Jeśli nie przetworzono żadnego inputu, zwróć neutralny sentyment
        if not input_processed or total_weight == 0:
            return 50.0
        
        # Oblicz średnią ważoną
        final_result = total_score / total_weight
        
        return final_result


# Alias dla kompatybilności wstecznej (deprecated)
AnfisEngine = FuzzyExpertSystem
