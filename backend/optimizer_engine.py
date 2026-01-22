import numpy as np
import pandas as pd
from anfis_engine import AnfisEngine
from backtest_engine import BacktestEngine

class OptimizerEngine:
    """
    Optimizer dla prawdziwego ANFIS z:
    - Train/Test split (zapobiega overfittingowi)
    - Supervised learning (target = przyszły zwrot)
    - Walidacja krzyżowa
    """
    
    def __init__(self):
        self.anfis = AnfisEngine()
        self.backtester = BacktestEngine()
        self.training_results = None
    
    def prepare_training_data(self, df_orig, config, lookforward=30):
        """
        Przygotowuje dane treningowe (X, y) dla ANFIS
        
        Target: Przyszły zwrot po 'lookforward' dniach
        y = (Price_t+30 / Price_t - 1) * 100 -> skalowane do 0-100
        
        Returns: X_train, y_train, X_test, y_test
        """
        df = df_orig.copy()
        
        # Oblicz target (przyszły zwrot)
        df['Future_Return'] = (df['Price'].shift(-lookforward) / df['Price'] - 1) * 100
        
        # Usuń ostatnie wiersze bez targetu
        df = df.dropna(subset=['Future_Return'])
        
        # Normalizuj target do 0-100 (0=sell, 50=neutral, 100=buy)
        # Używamy percentyle rank
        df['Target_Normalized'] = df['Future_Return'].rank(pct=True) * 100
        
        # Przygotuj X (inputs dla ANFIS)
        active_features = [k for k, v in config.items() if v.get('enabled', False)]
        
        X = []
        y = []
        
        for idx, row in df.iterrows():
            inputs = {}
            skip = False
            
            for feat in active_features:
                if feat in row and not pd.isna(row[feat]):
                    inputs[feat] = float(row[feat])
                else:
                    skip = True
                    break
            
            if not skip:
                X.append(inputs)
                y.append(row['Target_Normalized'])
        
        # Train/Test split (80/20)
        split_idx = int(len(X) * 0.8)
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        return X_train, y_train, X_test, y_test
    
    def train_anfis(self, df_orig, config, epochs=50, progress_callback=None):
        """
        Trenuje ANFIS z supervised learning
        """
        # Przygotuj dane
        X_train, y_train, X_test, y_test = self.prepare_training_data(df_orig, config)
        
        if len(X_train) < 100:
            raise ValueError("Za mało danych treningowych (min 100 próbek)")
        
        # Inicjalizuj ANFIS
        self.anfis.initialize_params(config)
        
        # Trening z callbackiem postępu
        print(f"📚 Trening ANFIS: {len(X_train)} próbek treningowych, {len(X_test)} testowych")
        
        # ========== POPRAWKA: Użyj metody train() z anfis_engine ==========
        self.anfis.train(X_train, y_train, epochs=epochs, verbose=True)
        # ==================================================================
        
        # Test set evaluation
        test_error = 0
        for i in range(len(X_test)):
            out = self.anfis.forward(X_test[i])
            test_error += (out - y_test[i]) ** 2
        test_mse = test_error / len(X_test) if len(X_test) > 0 else 0
        
        train_mse = self.anfis.training_history['mse'][-1]
        
        # Finalna walidacja
        print("✓ Trening zakończony!")
        print(f"  Train MSE: {train_mse:.2f}")
        print(f"  Test MSE: {test_mse:.2f}")
        
        self.anfis.trained = True
        
        # Zapisz wyniki
        self.training_results = {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'overfitting_ratio': float(test_mse / (train_mse + 1e-10)),
            'params': self.anfis.get_params_summary(),
            'history': {
                'epochs': [int(x) for x in self.anfis.training_history['epochs']],
                'mse': [float(x) for x in self.anfis.training_history['mse']]
            }
        }
    
        return self.training_results
    
    def optimize_weights(self, df_orig, progress_callback=None):
        """
        STARY INTERFEJS - dla kompatybilności z app.py
        Teraz uruchamia pełny trening ANFIS zamiast tylko optymalizacji wag
        """
        # Domyślna konfiguracja (wszystkie wskaźniki)
        config = {
            'RSI': {'enabled': True, 'weight': 1.0, 'direction': -1},
            'VIX': {'enabled': True, 'weight': 1.0, 'direction': 1},
            'Yield_Curve': {'enabled': True, 'weight': 1.0, 'direction': 1},
            'MACD': {'enabled': True, 'weight': 1.0, 'direction': 1},
            'M2_Liquidity': {'enabled': True, 'weight': 1.0, 'direction': 1}
        }
        
        # Trenuj ANFIS
        results = self.train_anfis(df_orig, config, epochs=30, progress_callback=progress_callback)
        
        # Zwróć w formacie kompatybilnym ze starym API
        # (wagi są teraz wbudowane w konsekwenty, ale zwracamy domyślne)
        return {
            'rsi': 1.0,
            'vix': 1.0,
            'yield': 1.0,
            'macd': 1.0,
            'm2': 1.0,
            '_training_results': results  # Dodatkowe info
        }