import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class AnfisEngine:
    def __init__(self):
        self.simulation = None

    def build_system(self):
        # --- ZMIENNE WEJŚCIOWE ---
        # Używamy zakresu 0-100 dla wszystkiego
        rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
        vix = ctrl.Antecedent(np.arange(0, 101, 1), 'vix')
        yield_curve = ctrl.Antecedent(np.arange(0, 101, 1), 'yield')

        # --- ZMIENNA WYJŚCIOWA ---
        # 0 = Ekstremalne Przewartościowanie (Szczyt bańki)
        # 100 = Ekstremalne Niedowartościowanie (Dno krachu)
        sentiment = ctrl.Consequent(np.arange(0, 101, 1), 'sentiment')

        # --- FUNKCJE PRZYNALEŻNOŚCI (GAUSS - PŁYNNE) ---
        # sigma = szerokość dzwonu (im większa, tym gładszy wykres)
        sigma_in = 20  
        
        # RSI (Technika)
        rsi['low'] = fuzz.gaussmf(rsi.universe, 20, sigma_in)
        rsi['medium'] = fuzz.gaussmf(rsi.universe, 50, sigma_in)
        rsi['high'] = fuzz.gaussmf(rsi.universe, 80, sigma_in)

        # VIX (Strach - Pamiętaj: High VIX = Strach = Dobrze dla kupującego)
        vix['low'] = fuzz.gaussmf(vix.universe, 20, sigma_in)      # Spokój (Ryzykowne)
        vix['medium'] = fuzz.gaussmf(vix.universe, 50, sigma_in)
        vix['high'] = fuzz.gaussmf(vix.universe, 80, sigma_in)     # Panika (Okazja)

        # Yield (Makro - Pamiętaj: High Yield Rank = Dobra gospodarka)
        yield_curve['low'] = fuzz.gaussmf(yield_curve.universe, 20, sigma_in)  # Recesja
        yield_curve['medium'] = fuzz.gaussmf(yield_curve.universe, 50, sigma_in)
        yield_curve['high'] = fuzz.gaussmf(yield_curve.universe, 80, sigma_in) # Wzrost

        # --- WYJŚCIE (5 POZIOMÓW) ---
        # Dzięki temu wynik nie skacze 0 -> 50 -> 100, ale np. 0 -> 25 -> 50...
        sigma_out = 15
        sentiment['strong_sell'] = fuzz.gaussmf(sentiment.universe, 10, sigma_out)
        sentiment['sell'] = fuzz.gaussmf(sentiment.universe, 30, sigma_out)
        sentiment['neutral'] = fuzz.gaussmf(sentiment.universe, 50, sigma_out)
        sentiment['buy'] = fuzz.gaussmf(sentiment.universe, 70, sigma_out)
        sentiment['strong_buy'] = fuzz.gaussmf(sentiment.universe, 90, sigma_out)

        # --- REGUŁY LOGICZNE (MATRIX) ---
        rules = []

        # GRUPA 1: EKSTREMA (Silne sygnały)
        
        # Super Okazja: Jest tanio (RSI Low) + Panika (VIX High) + Gospodarka OK (Yield High)
        rules.append(ctrl.Rule(rsi['low'] & vix['high'] & yield_curve['high'], sentiment['strong_buy']))
        
        # Krach/Recesja: Jest tanio (RSI Low), ale Gospodarka pada (Yield Low) -> Ostrożnie (tylko Buy, nie Strong)
        rules.append(ctrl.Rule(rsi['low'] & yield_curve['low'], sentiment['buy']))
        
        # Bańka Spekulacyjna: Jest drogo (RSI High) + Wszyscy wyluzowani (VIX Low) -> Uciekaj
        rules.append(ctrl.Rule(rsi['high'] & vix['low'], sentiment['strong_sell']))

        # GRUPA 2: STANY POŚREDNIE (Wygładzanie)

        # Trend boczny / Nuda: Wszystko średnie -> Neutral
        rules.append(ctrl.Rule(rsi['medium'] & vix['medium'], sentiment['neutral']))
        
        # Korekta w hossie: RSI spadło (Low), ale strachu nie ma (VIX Low) -> Małe kupno
        rules.append(ctrl.Rule(rsi['low'] & vix['low'], sentiment['buy']))
        
        # Przegrzanie w bessie: RSI urosło (High), ale nadal strach (VIX High) -> Mała sprzedaż
        rules.append(ctrl.Rule(rsi['high'] & vix['high'], sentiment['sell']))
        
        # Ostrzeżenie makro: Jeśli Yield spada (Recesja), nigdy nie dawaj Strong Buy, max Neutral
        rules.append(ctrl.Rule(yield_curve['low'] & rsi['medium'], sentiment['sell']))

        # Budowa systemu
        system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(system)

    def compute(self, rsi_val, vix_rank, yield_rank):
        if self.simulation is None:
            self.build_system()
            
        # Clip input values to ensure safety
        self.simulation.input['rsi'] = np.clip(rsi_val, 0, 100)
        self.simulation.input['vix'] = np.clip(vix_rank, 0, 100)
        self.simulation.input['yield'] = np.clip(yield_rank, 0, 100)
        
        try:
            # Defuzzification method 'centroid' (środek ciężkości)
            # To właśnie ta metoda w połączeniu z Gaussami daje płynny wynik
            self.simulation.compute()
            return self.simulation.output['sentiment']
        except Exception:
            # Fallback w razie błędu obliczeń (np. brak aktywnej reguły)
            return 50.0