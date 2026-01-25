"""
Weight Optimizer - Differential Evolution Tuner
===============================================

Ten moduł optymalizuje WAGI wskaźników w systemie FuzzyExpertSystem
metodą ewolucji różnicowej (Differential Evolution).

WAŻNE: To NIE jest uczenie ANFIS!

Co ten moduł robi:
- Szuka najlepszych WAG dla wskaźników (RSI, VIX, MACD, etc.)
- Używa algorytmu ewolucyjnego (meta-heurystyka)
- Maksymalizuje zwrot z backtestingu
- NIE modyfikuje funkcji przynależności ani reguł!

Czym to NIE jest:
- To NIE jest ANFIS (nie uczy funkcji przynależności)
- To NIE jest uczenie głębokie (brak gradientów)
- To NIE optymalizuje parametrów sieci neuronowej

Jak działa:
1. Generuje populację zestawów wag (np. 20 osobników)
2. Dla każdego zestawu: buduje FIS → backtest → oblicza zwrot
3. Selekcja + mutacja + krzyżowanie (ewolucja)
4. Powtarza przez N generacji
5. Zwraca najlepsze wagi

Ryzyko: OVERFITTING
- Wagi dopasowane do przeszłości mogą nie działać w przyszłości
- Zalecane: walidacja out-of-sample

Prawdziwe uczenie ANFIS: anfis_torch.py, anfis_ml_engine_v2.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from fuzzy_expert_system import FuzzyExpertSystem
from backtest_engine import BacktestEngine


class WeightOptimizer:
    """
    Optymalizator wag wskaźników metodą Differential Evolution.
    
    Szuka zestawu wag, który maksymalizuje zwrot z backtestingu.
    Używa FuzzyExpertSystem jako generatora sygnałów.
    
    Attributes:
        fis: Instancja FuzzyExpertSystem
        backtester: Instancja BacktestEngine
    
    Example:
        >>> optimizer = WeightOptimizer()
        >>> best_weights = optimizer.optimize_weights(df)
        >>> print(best_weights)
        {'rsi': 0.85, 'vix': 1.45, 'yield': 0.32, 'macd': 1.78, 'm2': 0.91}
    """
    
    def __init__(self):
        self.fis = FuzzyExpertSystem()
        self.backtester = BacktestEngine()

    def optimize_weights(self, df_orig, progress_callback=None):
        """
        Optymalizuje wagi wskaźników metodą Differential Evolution.
        
        Algorytm:
        1. Tworzy populację losowych zestawów wag
        2. Dla każdego zestawu oblicza "fitness" (zwrot z backtestu)
        3. Ewoluuje populację przez mutację i krzyżowanie
        4. Zwraca najlepszy znaleziony zestaw wag
        
        Args:
            df_orig: DataFrame z danymi (musi zawierać: RSI, VIX, Yield_Curve, MACD, M2_Liquidity, Price)
            progress_callback: Opcjonalna funkcja callback(percent) do raportowania postępu
        
        Returns:
            dict: Najlepsze wagi dla każdego wskaźnika, np.:
                {'rsi': 0.85, 'vix': 1.45, 'yield': 0.32, 'macd': 1.78, 'm2': 0.91}
        
        Warning:
            RYZYKO OVERFITTINGU! Wagi zoptymalizowane na danych historycznych
            mogą nie działać w przyszłości. Zalecana walidacja out-of-sample.
        """
        df = df_orig.copy()
        
        # Przygotowanie danych - normalizacja do zakresu 0-100
        inputs_data = {}
        mapping = [
            ('RSI', 'rsi'), 
            ('VIX', 'vix'), 
            ('Yield_Curve', 'yield'), 
            ('MACD', 'macd'), 
            ('M2_Liquidity', 'm2')
        ]
        
        for col, key in mapping:
            if col in df.columns:
                series = df[col]
                # Normalizacja min-max do zakresu 0-100
                min_val = series.min()
                max_val = series.max()
                if max_val > min_val:
                    normalized = ((series - min_val) / (max_val - min_val) * 100).fillna(50)
                else:
                    normalized = pd.Series([50] * len(series))
                inputs_data[key] = normalized.values
            else:
                # Brak kolumny - użyj neutralnej wartości
                inputs_data[key] = np.full(len(df), 50)
        
        def objective(weights):
            """
            Funkcja celu do minimalizacji.
            
            Zwraca UJEMNY zwrot z backtestu (bo algorytm minimalizuje).
            """
            # Konfiguracja FIS z testowanymi wagami
            config = {
                'rsi':   {'enabled': True, 'weight': weights[0], 'direction': -1},
                'vix':   {'enabled': True, 'weight': weights[1], 'direction': -1},
                'yield': {'enabled': True, 'weight': weights[2], 'direction': 1},
                'macd':  {'enabled': True, 'weight': weights[3], 'direction': 1},
                'm2':    {'enabled': True, 'weight': weights[4], 'direction': 1},
            }
            self.fis.build_system(config)
            
            # Oblicz oscylator dla każdego N-tego dnia (przyspieszenie)
            scores = []
            step = 5  # Co 5 dni
            indices = range(0, len(df), step)
            
            for i in indices:
                row_inputs = {k: v[i] for k, v in inputs_data.items()}
                scores.append(self.fis.compute(row_inputs))
            
            # Interpolacja do pełnej długości
            full_scores = np.interp(np.arange(len(df)), list(indices), scores)
            df['Sentiment_Oscillator'] = full_scores
            df['Date'] = df.index
            
            # Backtest
            result = self.backtester.run(df, initial_capital=10000)
            
            # Zwróć UJEMNY zwrot (minimalizacja → maksymalizacja zwrotu)
            return -result['total_return']

        print("🧬 Rozpoczynanie optymalizacji ewolucyjnej...")
        
        # Parametry algorytmu
        max_iter = 10  # Liczba generacji
        current_iter = 0

        def callback_fn(xk, convergence=None):
            """Callback wywoływany po każdej generacji."""
            nonlocal current_iter
            current_iter += 1
            if progress_callback:
                progress = min(int((current_iter / max_iter) * 100), 99)
                progress_callback(progress)

        # Granice wag: 0.0 - 2.0 dla każdego wskaźnika
        bounds = [(0, 2.0)] * 5
        
        # Uruchomienie Differential Evolution
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=4,      # 4 osobniki na wymiar = 20 osobników
            seed=42,        # Dla powtarzalności
            callback=callback_fn
        )
        
        best_weights = result.x
        
        print(f"✅ Optymalizacja zakończona. Najlepszy zwrot: {-result.fun:.2f}%")
        
        return {
            'rsi':   round(best_weights[0], 2),
            'vix':   round(best_weights[1], 2),
            'yield': round(best_weights[2], 2),
            'macd':  round(best_weights[3], 2),
            'm2':    round(best_weights[4], 2)
        }


# Alias dla kompatybilności wstecznej (deprecated)
OptimizerEngine = WeightOptimizer
