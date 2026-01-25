import numpy as np
from scipy.optimize import differential_evolution
from anfis_engine import AnfisEngine
from backtest_engine import BacktestEngine

class OptimizerEngine:
    def __init__(self):
        self.anfis = AnfisEngine()
        self.backtester = BacktestEngine()

    # Dodajemy argument 'progress_callback'
    def optimize_weights(self, df_orig, progress_callback=None):
        df = df_orig.copy()
        
        # Przygotowanie danych (Normalizacja)
        inputs_data = {}
        mapping = [('RSI', 'rsi'), ('VIX', 'vix'), ('Yield_Curve', 'yield'), ('MACD', 'macd'), ('M2_Liquidity', 'm2')]
        
        for col, key in mapping:
             if col in df.columns:
                 series = df[col]
                 normalized = ((series - series.min()) / (series.max() - series.min()) * 100).fillna(50)
                 inputs_data[key] = normalized.values
             else:
                 inputs_data[key] = np.full(len(df), 50)
        
        # Funkcja celu
        def objective(weights):
            config = {
                'rsi':   {'enabled': True, 'weight': weights[0]},
                'vix':   {'enabled': True, 'weight': weights[1]},
                'yield': {'enabled': True, 'weight': weights[2]},
                'macd':  {'enabled': True, 'weight': weights[3]},
                'm2':    {'enabled': True, 'weight': weights[4]},
            }
            self.anfis.build_system(config)
            
            scores = []
            step = 5
            indices = range(0, len(df), step)
            
            for i in indices: 
                row_inputs = {k: v[i] for k, v in inputs_data.items()}
                scores.append(self.anfis.compute(row_inputs))
            
            full_scores = np.interp(np.arange(len(df)), list(indices), scores)
            df['Sentiment_Oscillator'] = full_scores
            df['Date'] = df.index
            
            res = self.backtester.run(df, initial_capital=10000)
            return -res['total_return']

        print("Rozpoczynanie AI...")
        
        # Ustawienia algorytmu
        max_iter = 10 # ZwiÄ™kszamy trochÄ™ liczbÄ™ iteracji dla lepszego efektu
        current_iter = 0

        # Funkcja wywoÅ‚ywana po kaÅ¼dej generacji algorytmu
        def callback_fn(xk, convergence=None):
            nonlocal current_iter
            current_iter += 1
            if progress_callback:
                # Obliczamy procent (zabezpieczenie przed wyjÅ›ciem poza 100%)
                progress = min(int((current_iter / max_iter) * 100), 99)
                progress_callback(progress)

        bounds = [(0, 2.0)] * 5
        
        # Uruchomienie z callbackiem
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=4, 
            seed=42, 
            callback=callback_fn
        )
        
        best_weights = result.x
        
        return {
            'rsi':   round(best_weights[0], 2),
            'vix':   round(best_weights[1], 2),
            'yield': round(best_weights[2], 2),
            'macd':  round(best_weights[3], 2),
            'm2':    round(best_weights[4], 2)
        }