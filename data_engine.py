import pandas as pd
import yfinance as yf
from fredapi import Fred
import numpy as np

class DataEngine:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)

    def get_market_data(self, ticker, period="max"):
        print(f"Pobieranie danych giełdowych dla {ticker}...")
        df = yf.download(ticker, period=period, interval="1d")
        
        # Standaryzacja kolumn (obsługa nowych wersji yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs('Close', level=0, axis=1)
            except:
                df = df['Close']
        
        # Upewniamy się, że mamy Series lub DataFrame z jedną kolumną
        if isinstance(df, pd.Series):
            df = df.to_frame(name='Price')
        else:
            df = df[['Close']] if 'Close' in df.columns else df
            df.columns = ['Price']

        # --- OBLICZANIE RSI (Technika) ---
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def get_macro_data(self):
        """Pobiera VIX i Krzywą Dochodowości"""
        print("Pobieranie danych makro (VIX + Yield Curve)...")
        
        # T10Y2Y: Różnica między obligacjami 10L a 2L (Poniżej 0 = Recesja)
        yield_curve = self.fred.get_series('T10Y2Y')
        
        # VIXCLS: Indeks strachu (Wysoko = Panika)
        vix = self.fred.get_series('VIXCLS')
        
        macro_df = pd.DataFrame({
            'Yield_Curve': yield_curve,
            'VIX': vix
        })
        return macro_df

    def prepare_dataset(self, ticker):
        market_df = self.get_market_data(ticker)
        macro_df = self.get_macro_data()
        
        # Łączenie danych (Left join do cen akcji)
        df = market_df.join(macro_df, how='left')
        
        # Wypełnianie danych (VIX i Yield są dzienne, ale mają inne święta)
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        
        return df