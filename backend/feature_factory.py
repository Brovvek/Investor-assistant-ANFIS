import pandas as pd
import numpy as np

class FeatureFactory:
    @staticmethod
    def generate_features(df_input):
        """
        Generuje szeroki wachlarz wskaźników technicznych i statystycznych.
        """
        df = df_input.copy()
        
        # Okna czasowe
        windows = [5, 10, 20, 50, 100]
        
        # Lista kolumn bazowych - DODALIŚMY MACD
        base_cols = ['Price']
        possible_cols = ['RSI', 'VIX', 'Yield_Curve', 'M2_Liquidity', 'MACD']
        
        for col in possible_cols:
            if col in df.columns:
                base_cols.append(col)

        for col in base_cols:
            for w in windows:
                # 1. Momentum (ROC)
                df[f"{col}_ROC_{w}"] = df[col].pct_change(periods=w)

                # 2. Zmienność (Volatility)
                df[f"{col}_Volat_{w}"] = df[col].rolling(window=w).std()
                
                # 3. Odchylenie od średniej (Trend)
                sma = df[col].rolling(window=w).mean()
                # Unikamy dzielenia przez zero (epsilon)
                df[f"{col}_DistSMA_{w}"] = (df[col] - sma) / sma.replace(0, 0.0001)

        df.dropna(inplace=True)
        return df

    @staticmethod
    def select_diverse_top_features(df, target_col='Target', top_n=5, max_correlation=0.6):
        """
        Wybiera N najlepszych cech, unikając duplikatów (korelacji między sobą).
        """
        # Obliczamy korelacje z Targetem
        all_correlations = df.corr(method='spearman')[target_col].drop(target_col)
        
        # Sortujemy od najsilniejszej
        sorted_features = all_correlations.abs().sort_values(ascending=False).index.tolist()
        
        selected_features = []
        selected_corrs = {}

        for feature in sorted_features:
            if len(selected_features) >= top_n:
                break
            
            # Pomijamy kolumny techniczne
            if feature in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Date', 'Price']:
                continue

            # Filtr redundancji
            is_redundant = False
            for selected in selected_features:
                col_corr = df[feature].corr(df[selected])
                if abs(col_corr) > max_correlation:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_features.append(feature)
                selected_corrs[feature] = all_correlations[feature]

        return selected_features, selected_corrs