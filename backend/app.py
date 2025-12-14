from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from data_engine import DataEngine
from anfis_engine import AnfisEngine
from backtest_engine import BacktestEngine
from optimizer_engine import OptimizerEngine
import pandas as pd
import json

# --- IMPORT KONFIGURACJI ---
# Dzięki temu klucz jest bezpieczny w innym pliku
try:
    from config import FRED_API_KEY
except ImportError:
    print("BŁĄD: Nie znaleziono pliku config.py lub klucza FRED_API_KEY!")
    FRED_API_KEY = None

app = Flask(__name__)
CORS(app)

# --- INICJALIZACJA SILNIKÓW ---
data_engine = DataEngine(api_key=FRED_API_KEY) 
anfis_engine = AnfisEngine()
backtester = BacktestEngine()
optimizer = OptimizerEngine()

# --- NOWA FUNKCJA NORMALIZACJI (Percentile Rank) ---
# Naprawia problem "płaskich" wykresów dla M2 i VIX
def normalize_rank(series, window=504):
    """
    Zamienia wartości na percentyle (0-100) w oknie kroczącym (ok. 2 lata).
    Dzięki temu wskaźniki makro (M2, VIX) stają się dynamiczne dla ANFIS.
    """
    if series.empty: return series
    # rank(pct=True) zwraca 0.0-1.0, mnożymy * 100
    return series.rolling(window).rank(pct=True) * 100

# --- ENDPOINTY ---

@app.route('/api/tickers', methods=['GET'])
def get_available_tickers():
    tickers = [
        {"symbol": "^GSPC", "name": "S&P 500 Index (USA)"},
        {"symbol": "^NDX", "name": "Nasdaq 100 Index (USA)"},
        {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
        {"symbol": "BTC-USD", "name": "Bitcoin / USD"},
        {"symbol": "GLD", "name": "Złoto (Gold Shares)"},
        {"symbol": "TLT", "name": "Obligacje USA 20Y+"},
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corp."}
    ]
    return jsonify(tickers)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    config = req_data.get('config', {})
    
    try:
        df = data_engine.prepare_dataset(ticker)
        if df.empty: return jsonify({"error": "Brak danych"}), 400

        # --- NORMALIZACJA RANGOWA (Dynamiczna) ---
        # Używamy okna 504 dni (2 lata giełdowe) do oceny "czy jest drogo/tanio"
        window = 504
        
        # Obliczamy rangi (0-100) dla wszystkich wskaźników
        # fillna(50) zabezpiecza początek wykresu przed błędami
        df['RSI_Rank'] = normalize_rank(df['RSI'], window).fillna(50)
        df['VIX_Rank'] = normalize_rank(df['VIX'], window).fillna(50)
        df['Yield_Rank'] = normalize_rank(df['Yield_Curve'], window).fillna(50)
        
        # MACD i M2
        if 'MACD' in df.columns: 
            df['MACD_Rank'] = normalize_rank(df['MACD'], window).fillna(50)
        else: 
            df['MACD_Rank'] = 50
            
        if 'M2_Liquidity' in df.columns: 
            df['M2_Rank'] = normalize_rank(df['M2_Liquidity'], window).fillna(50)
        else: 
            df['M2_Rank'] = 50

        # Usuwamy puste wiersze powstałe po rolling window
        df.dropna(inplace=True)

        # Budowa systemu ANFIS
        anfis_engine.build_system(config)
        
        # Obliczenia
        oscillator_values = []
        df_analysis = df.tail(1260).copy() # Ostatnie 5 lat do wyświetlenia

        for index, row in df_analysis.iterrows():
            inputs = {
                # Ważne: Przekazujemy znormalizowane rangi!
                'rsi': row['RSI_Rank'],
                'vix': row['VIX_Rank'],
                'yield': row['Yield_Rank'],
                'macd': row['MACD_Rank'],
                'm2': row['M2_Rank']
            }
            score = anfis_engine.compute(inputs)
            oscillator_values.append(score)
            
        df_analysis['Sentiment_Oscillator'] = oscillator_values
        
        # Formatowanie dla Frontendu
        df_analysis.index.name = 'Date'
        df_analysis.reset_index(inplace=True)
        df_analysis['Date'] = df_analysis['Date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(df_analysis.to_dict(orient='list'))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    config = req_data.get('config', {})
    
    # Parametry Strategii
    buy_thr = float(req_data.get('buyThreshold', 70))
    sell_thr = float(req_data.get('sellThreshold', 30))
    stop_loss = float(req_data.get('stopLoss', 5)) / 100.0
    take_profit = float(req_data.get('takeProfit', 0)) / 100.0
    trailing_stop = req_data.get('trailingStop', False)
    
    try:
        # 1. Oblicz ANFIS (powtórzenie logiki z analyze z nową normalizacją)
        df = data_engine.prepare_dataset(ticker)
        
        window = 504
        df['RSI_Rank'] = normalize_rank(df['RSI'], window).fillna(50)
        df['VIX_Rank'] = normalize_rank(df['VIX'], window).fillna(50)
        df['Yield_Rank'] = normalize_rank(df['Yield_Curve'], window).fillna(50)
        
        if 'MACD' in df.columns: df['MACD_Rank'] = normalize_rank(df['MACD'], window).fillna(50)
        else: df['MACD_Rank'] = 50
        
        if 'M2_Liquidity' in df.columns: df['M2_Rank'] = normalize_rank(df['M2_Liquidity'], window).fillna(50)
        else: df['M2_Rank'] = 50
        
        df.dropna(inplace=True)
        
        anfis_engine.build_system(config)
        
        oscillator_values = []
        df_analysis = df.tail(1260).copy()
        
        for index, row in df_analysis.iterrows():
            inputs = {
                'rsi': row['RSI_Rank'],
                'vix': row['VIX_Rank'],
                'yield': row['Yield_Rank'],
                'macd': row['MACD_Rank'],
                'm2': row['M2_Rank']
            }
            oscillator_values.append(anfis_engine.compute(inputs))
            
        df_analysis['Sentiment_Oscillator'] = oscillator_values
        df_analysis.index.name = 'Date'
        df_analysis.reset_index(inplace=True)
        df_analysis['Date'] = df_analysis['Date'].dt.strftime('%Y-%m-%d')
        
        # 2. Backtest (z obsługą Stop Loss i Prowizji)
        result = backtester.run(
            df_analysis, 
            buy_threshold=buy_thr, 
            sell_threshold=sell_thr,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            use_trailing_stop=trailing_stop,
            fee_pct=0.001 
        )
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/correlations', methods=['POST'])
def calculate_correlations():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    
    try:
        df = data_engine.prepare_dataset(ticker)
        # Przewidywanie zwrotu za 30 dni
        df['Future_Return'] = df['Price'].shift(-30) / df['Price'] - 1
        df.dropna(inplace=True)
        
        features = ['RSI', 'VIX', 'Yield_Curve', 'M2_Liquidity', 'Inflation_CPI', 'MACD']
        correlations = {}
        
        for feature in features:
            if feature in df.columns:
                val = df[feature].corr(df['Future_Return'], method='spearman')
                correlations[feature] = round(val, 4)
                
        return jsonify(correlations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    
    # STREAMING: Używamy stream_with_context dla paska postępu
    @stream_with_context
    def generate():
        try:
            df = data_engine.prepare_dataset(ticker)
            
            # Callback wysyłający postęp do frontendu
            def on_progress(percent):
                yield json.dumps({"status": "progress", "value": percent}) + "\n"

            # Uruchamiamy AI
            best_weights = optimizer.optimize_weights(df, progress_callback=on_progress)
            
            # Wynik końcowy
            yield json.dumps({"status": "done", "result": best_weights}) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return Response(generate(), mimetype='application/x-json-stream')

if __name__ == '__main__':
    app.run(debug=True)