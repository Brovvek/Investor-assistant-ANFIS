import pandas as pd
import yfinance as yf
from fredapi import Fred

class DataEngine:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)

    def get_market_data(self, ticker, period="5y"):
        print(f"Pobieranie danych giełdowych dla {ticker}...")
        df = yf.download(ticker, period=period, interval="1d")
        if df.empty:
            raise ValueError("Nie znaleziono danych dla podanego tickera.")
        
        # Obsługa MultiIndex w nowym yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', level=0, axis=1) if 'Close' in df.columns.levels[0] else df['Close']
        else:
            df = df[['Close']]
            
        # Upewnienie się, że mamy jedną kolumnę o nazwie 'Price'
        if isinstance(df, pd.Series):
            df = df.to_frame(name='Price')
        else:
            df.columns = ['Price']
            
        return df

    def get_macro_series(self, series_id, transformation='raw'):
        print(f"Pobieranie makro: {series_id} ({transformation})...")
        try:
            data = self.fred.get_series(series_id)
            if transformation == 'yoy':
                data = data.pct_change(periods=12) * 100
            elif transformation == 'mom':
                data = data.pct_change(periods=1) * 100
            return data
        except Exception as e:
            print(f"Błąd pobierania {series_id}: {e}")
            return pd.Series()

    def prepare_dataset(self, ticker, macro_config):
        # 1. Dane giełdowe
        market_df = self.get_market_data(ticker)
        
        # 2. Dane makro
        macro_df = pd.DataFrame()
        for sid, transform in macro_config.items():
            s_data = self.get_macro_series(sid, transform)
            macro_df[sid] = s_data

        # 3. Łączenie i synchronizacja
        # Dołączamy makro do dat giełdowych
        combined = market_df.join(macro_df, how='outer')
        
        # Wypełnianie braków (dane makro są miesięczne, giełda dzienna)
        combined.fillna(method='ffill', inplace=True)
        combined.dropna(inplace=True) # Usuwamy początkowe braki
        
        return combined