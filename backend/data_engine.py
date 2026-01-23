import pandas as pd
import yfinance as yf
import numpy as np
import requests
import sys

# Wymuszenie kodowania UTF-8 dla konsoli
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

class DataEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def _fetch_fred_direct(self, series_id):
        """
        Pobiera dane bezpoÅ›rednio z URL API FRED, omijajÄ…c bibliotekÄ™ fredapi
        i problemy z polskimi znakami w Å›cieÅ¼kach systemowych.
        """
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status() # ZgÅ‚oÅ› bÅ‚Ä…d jeÅ›li status != 200
            data = response.json()
            
            # Parsowanie JSON do DataFrame
            observations = data.get('observations', [])
            if not observations:
                return pd.Series(dtype=float)

            df = pd.DataFrame(observations)
            
            # Konwersja danych
            df['date'] = pd.to_datetime(df['date'])
            # '.' w FRED oznacza brak danych, zamieniamy na NaN
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            df.set_index('date', inplace=True)
            return df['value']
            
        except Exception as e:
            print(f"âš ï¸ BÅ‚Ä…d pobierania {series_id} (Direct): {e}")
            return pd.Series(dtype=float)

    def get_market_data(self, ticker, period="max"):
        print(f"Pobieranie danych gieÅ‚dowych dla {ticker}...")
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False)
        except Exception as e:
            print(f"BÅ‚Ä…d yfinance: {e}")
            return pd.DataFrame()
        
        if df.empty: return pd.DataFrame()

        # ObsÅ‚uga kolumn (MultiIndex fix)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs('Close', level=0, axis=1)
            except: 
                if 'Close' in df.columns: df = df[['Close']]
                else: df = df.iloc[:, 0].to_frame()
        
        if isinstance(df, pd.Series): df = df.to_frame(name='Price')
        else:
            df = df.rename(columns={df.columns[0]: 'Price'})
            df = df[['Price']]

        # --- WSKAÅ¹NIKI TECHNICZNE ---
        try:
            # 1. RSI
            delta = df['Price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # 2. MACD HISTOGRAM (Zmiana!)
            # MACD Line
            exp12 = df['Price'].ewm(span=12, adjust=False).mean()
            exp26 = df['Price'].ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            # Signal Line
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            # Histogram (To nas interesuje - jest dynamiczne)
            df['MACD'] = macd_line - signal_line 

        except Exception as e:
            print(f"BÅ‚Ä…d wskaÅºnikÃ³w: {e}")

        return df
    def get_macro_data(self):
        print("Pobieranie danych makro (Direct API)...")
        
        # UÅ¼ywamy nowej metody _fetch_fred_direct zamiast biblioteki fredapi
        
        # 1. Yield Curve (T10Y2Y)
        yield_curve = self._fetch_fred_direct('T10Y2Y')
        
        # 2. VIX (VIXCLS)
        vix = self._fetch_fred_direct('VIXCLS')
        
        # 3. M2 Money Supply (M2SL) -> YoY
        m2 = self._fetch_fred_direct('M2SL')
        # M2 jest miesięczne, ale yfinance dzienne. fillna załatwi sprawę później.
        m2_yoy = m2.pct_change(periods=12, fill_method=None) * 100 

        # 4. Inflacja CPI (CPIAUCSL) -> YoY
        cpi = self._fetch_fred_direct('CPIAUCSL')
        cpi_yoy = cpi.pct_change(periods=12, fill_method=None) * 100

        # Tworzenie DataFrame
        macro_df = pd.DataFrame({
            'Yield_Curve': yield_curve,
            'VIX': vix,
            'M2_Liquidity': m2_yoy,
            'Inflation_CPI': cpi_yoy
        })
        
        return macro_df

    def prepare_dataset(self, ticker):
        market_df = self.get_market_data(ticker)
        
        if market_df.empty:
            return pd.DataFrame()

        macro_df = self.get_macro_data()
        
        # ÅÄ…czenie (Left Join do cen akcji)
        df = market_df.join(macro_df, how='left')
        
        # WypeÅ‚nianie danych makro (ffill) - rozciÄ…gamy dane miesiÄ™czne na dzienne
        df.ffill(inplace=True)
        
        # Zabezpieczenie: JeÅ›li API nie zadziaÅ‚aÅ‚o, wstawiamy zera, Å¼eby aplikacja nie padÅ‚a
        expected_cols = ['Yield_Curve', 'VIX', 'M2_Liquidity', 'Inflation_CPI']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0 
        
        # Usuwamy puste wiersze na poczÄ…tku historii
        df.dropna(inplace=True)
        
        return df