import pandas as pd
import yfinance as yf
import requests
import sys
from functools import lru_cache
import datetime

# Wymuszenie kodowania UTF-8 dla konsoli (naprawia błędy przy printowaniu polskich znaków)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

class DataEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def _fetch_fred_direct(self, series_id):
        """
        Pobiera dane bezpośrednio z URL API FRED.
        """
        if not self.api_key:
            print(f"⚠️ Brak klucza API FRED. Pomijam pobieranie {series_id}.")
            return pd.Series(dtype=float)

        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            observations = data.get('observations', [])
            if not observations:
                return pd.Series(dtype=float)

            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            df.set_index('date', inplace=True)
            return df['value']
            
        except Exception as e:
            print(f"⚠️ Błąd pobierania {series_id} (Direct): {e}")
            return pd.Series(dtype=float)

    # --- CACHING ---
    # Zapamiętujemy wyniki dla 32 ostatnich zapytań.
    # Argument 'date_str' służy do inwalidacji cache'u (codziennie nowe dane).
    
    @lru_cache(maxsize=32)
    def _get_market_data_cached(self, ticker, date_str):
        print(f"⚡ POBIERANIE DANYCH GIEŁDOWYCH: {ticker} ({date_str})")
        try:
            # FIX: auto_adjust=False usuwa FutureWarning z yfinance
            df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=False)
        except Exception as e:
            print(f"Błąd yfinance: {e}")
            return pd.DataFrame()
        
        if df.empty: return pd.DataFrame()

        # Obsługa MultiIndex (częsty problem w nowym yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            try: 
                # Próbujemy wyciągnąć 'Close' lub 'Adj Close'
                if 'Adj Close' in df.columns.get_level_values(0):
                    df = df.xs('Adj Close', level=0, axis=1)
                elif 'Close' in df.columns.get_level_values(0):
                    df = df.xs('Close', level=0, axis=1)
                else:
                    # Fallback: bierzemy pierwszą kolumnę numeryczną
                    df = df.iloc[:, 0].to_frame()
            except: 
                df = df.iloc[:, 0].to_frame()
        
        # Upewniamy się, że mamy kolumnę 'Price'
        if isinstance(df, pd.Series): 
            df = df.to_frame(name='Price')
        else:
            # Jeśli df ma jedną kolumnę, nazywamy ją Price
            if len(df.columns) == 1:
                df.columns = ['Price']
            elif 'Close' in df.columns:
                df = df[['Close']].rename(columns={'Close': 'Price'})
            elif 'Adj Close' in df.columns:
                df = df[['Adj Close']].rename(columns={'Adj Close': 'Price'})
            else:
                df = df.rename(columns={df.columns[0]: 'Price'})
                df = df[['Price']]
            
        return df

    @lru_cache(maxsize=1) # Cache dla Makro (wspólny dla wszystkich tickerów)
    def _get_macro_data_cached(self, date_str):
        print(f"⚡ POBIERANIE DANYCH MAKRO ({date_str})")
        
        # Pobieranie surowych serii
        yield_curve = self._fetch_fred_direct('T10Y2Y')
        vix = self._fetch_fred_direct('VIXCLS')
        m2 = self._fetch_fred_direct('M2SL')
        cpi = self._fetch_fred_direct('CPIAUCSL')
        
        # Obliczenia dynamiki (YoY)
        # FIX: fill_method=None usuwa FutureWarning z pandas
        m2_yoy = m2.pct_change(periods=12, fill_method=None) * 100 
        cpi_yoy = cpi.pct_change(periods=12, fill_method=None) * 100

        macro_df = pd.DataFrame({
            'Yield_Curve': yield_curve,
            'VIX': vix,
            'M2_Liquidity': m2_yoy,
            'Inflation_CPI': cpi_yoy
        })
        return macro_df

    def prepare_dataset(self, ticker):
        """
        Główna funkcja budująca dataset.
        Łączy dane rynkowe i makro, oraz liczy podstawowe wskaźniki (RSI, MACD)
        wymagane do wizualizacji na wykresie.
        """
        # Data jako klucz cache
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        # 1. Pobierz dane giełdowe
        market_df = self._get_market_data_cached(ticker, today_str).copy()
        if market_df.empty:
            return pd.DataFrame()

        # 2. Pobierz dane makro
        macro_df = self._get_macro_data_cached(today_str).copy()
        
        # 3. Łączenie (Left Join do dat giełdowych)
        df = market_df.join(macro_df, how='left')
        
        # 4. Wypełnianie braków (Forward Fill)
        # Makroekonomia wychodzi rzadko (miesięcznie), więc powielamy ostatnią wartość
        df.ffill(inplace=True)
        
        # Jeśli na początku brakuje danych makro, wypełniamy zerami, żeby nie tracić wierszy
        for col in ['Yield_Curve', 'VIX', 'M2_Liquidity', 'Inflation_CPI']:
            if col not in df.columns: df[col] = 0.0
        df.fillna(0, inplace=True)
            
        # 5. Obliczanie podstawowych wskaźników technicznych
        # (Są potrzebne do wykresu, nawet jeśli FeatureFactory robi swoje)
        try:
            # RSI
            delta = df['Price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 0.001) # Unikamy dzielenia przez 0
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            exp12 = df['Price'].ewm(span=12, adjust=False).mean()
            exp26 = df['Price'].ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            # Histogram
            df['MACD'] = macd_line - signal_line
        except Exception as e:
            print(f"Błąd obliczania wskaźników bazowych: {e}")
            pass
            
        return df