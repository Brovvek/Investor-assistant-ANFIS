from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from data_engine import DataEngine
from anfis_engine import AnfisEngine
from backtest_engine import BacktestEngine
from optimizer_engine import OptimizerEngine
from feature_factory import FeatureFactory
from neuro_engine import NeuroEngine
import pandas as pd
import numpy as np
import json
import sys

# Wymuszenie UTF-8 w konsoli (Windows fix)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

try:
    from config import FRED_API_KEY
except ImportError:
    FRED_API_KEY = None

app = Flask(__name__)
CORS(app)

# Inicjalizacja silników
data_engine = DataEngine(api_key=FRED_API_KEY) 
anfis_engine = AnfisEngine()
backtester = BacktestEngine()
optimizer = OptimizerEngine()
neuro_engine = NeuroEngine()

# --- FUNKCJE POMOCNICZE ---

def normalize_rank(series, window=252):
    if series.empty: return series
    return series.rolling(window, min_periods=1).rank(pct=True) * 100

def prepare_data_with_features(ticker):
    df_raw = data_engine.prepare_dataset(ticker)
    if df_raw.empty: return pd.DataFrame()
    
    df = FeatureFactory.generate_features(df_raw)
    
    # Normalizacja
    cols_to_check = [c for c in df.columns if c not in ['Date', 'Price', 'Open', 'High', 'Low', 'Close', 'Adj Close']]
    window = 252 
    for col in cols_to_check:
        if col == 'RSI' or 'RSI_' in col and 'Rank' not in col:
            df[col] = df[col].fillna(50)
        else:
            df[col] = normalize_rank(df[col], window).fillna(50)
    
    # Wypełnianie braków (Robust)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.fillna(50, inplace=True)
    
    return df

# --- ENDPOINTY ---

@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    return jsonify([
        {"symbol": "^GSPC", "name": "S&P 500 (USA)"},
        {"symbol": "^NDX", "name": "Nasdaq 100 (USA)"},
        {"symbol": "^DJI", "name": "Dow Jones Ind."},
        {"symbol": "BTC-USD", "name": "Bitcoin / USD"},
        {"symbol": "ETH-USD", "name": "Ethereum / USD"},
        {"symbol": "GLD", "name": "Złoto (Gold)"},
        {"symbol": "TLT", "name": "Obligacje USA 20Y+"},
        {"symbol": "NVDA", "name": "NVIDIA Corp."},
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corp."},
        {"symbol": "EURUSD=X", "name": "EUR/USD"}
    ])

@app.route('/api/auto_strategy', methods=['POST'])
def auto_strategy():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    try:
        df_raw = data_engine.prepare_dataset(ticker)
        df = FeatureFactory.generate_features(df_raw)
        df['Target'] = df['Price'].shift(-30) / df['Price'] - 1
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        
        top_features, top_corrs = FeatureFactory.select_diverse_top_features(
            df, target_col='Target', top_n=5, max_correlation=0.6
        )
        
        new_config = {}
        for feat in top_features:
            corr_val = top_corrs[feat]
            new_config[feat] = {
                'enabled': True,
                'weight': round(abs(corr_val) * 3, 1),
                'direction': 1 if corr_val > 0 else -1
            }
        return jsonify(new_config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/correlations', methods=['POST'])
def calculate_correlations():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    try:
        df_raw = data_engine.prepare_dataset(ticker)
        df = FeatureFactory.generate_features(df_raw)
        df['Target_Return'] = df['Price'].shift(-30) / df['Price'] - 1
        df.dropna(inplace=True) 
        cols_to_analyze = [c for c in df.columns if c not in ['Target_Return', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Price']]
        data_matrix = df[cols_to_analyze].join(df['Target_Return'])
        
        corr_p = data_matrix.corr(method='pearson')['Target_Return'].drop('Target_Return')
        corr_s = data_matrix.corr(method='spearman')['Target_Return'].drop('Target_Return')
        corr_k = data_matrix.corr(method='kendall')['Target_Return'].drop('Target_Return')
        
        avg_strength = (corr_p.abs() + corr_s.abs() + corr_k.abs()) / 3
        top_features = avg_strength.sort_values(ascending=False).head(50).index.tolist()
        
        final_data = {}
        for feature in top_features:
            final_data[feature] = {
                'pearson': round(corr_p.get(feature, 0), 4),
                'spearman': round(corr_s.get(feature, 0), 4),
                'kendall': round(corr_k.get(feature, 0), 4)
            }
        return jsonify(final_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    config = req_data.get('config', {})
    try:
        df = prepare_data_with_features(ticker)
        if df.empty: return jsonify({"error": "Brak danych"}), 400

        anfis_engine.build_system(config)
        oscillator_values = []
        df_analysis = df.tail(1260).copy()
        active_features = [k for k, v in config.items() if v.get('enabled')]

        for index, row in df_analysis.iterrows():
            inputs = {feat: row[feat] for feat in active_features if feat in row}
            score = anfis_engine.compute(inputs)
            oscillator_values.append(score)
            
        df_analysis['Sentiment_Oscillator'] = oscillator_values
        df_analysis.index.name = 'Date'
        df_analysis.reset_index(inplace=True)
        df_analysis['Date'] = df_analysis['Date'].dt.strftime('%Y-%m-%d')
        return jsonify(df_analysis.to_dict(orient='list'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    config = req_data.get('config', {})
    buy_thr = float(req_data.get('buyThreshold', 70))
    sell_thr = float(req_data.get('sellThreshold', 30))
    stop_loss = float(req_data.get('stopLoss', 5)) / 100.0
    take_profit = float(req_data.get('takeProfit', 0)) / 100.0
    trailing_stop = req_data.get('trailingStop', False)
    
    try:
        df = prepare_data_with_features(ticker)
        anfis_engine.build_system(config)
        oscillator_values = []
        df_analysis = df.tail(1260).copy()
        active_features = [k for k, v in config.items() if v.get('enabled')]
        
        for index, row in df_analysis.iterrows():
            inputs = {feat: row[feat] for feat in active_features if feat in row}
            oscillator_values.append(anfis_engine.compute(inputs))
            
        df_analysis['Sentiment_Oscillator'] = oscillator_values
        df_analysis.index.name = 'Date'
        df_analysis.reset_index(inplace=True)
        df_analysis['Date'] = df_analysis['Date'].dt.strftime('%Y-%m-%d')
        
        result = backtester.run(
            df_analysis, buy_threshold=buy_thr, sell_threshold=sell_thr,
            stop_loss_pct=stop_loss, take_profit_pct=take_profit,
            use_trailing_stop=trailing_stop, fee_pct=0.001
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NOWY ENDPOINT STRUMIENIOWY ---
@app.route('/api/train_neuro', methods=['POST'])
def train_neuro():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    features = req_data.get('features', []) 
    epochs = int(req_data.get('epochs', 200))
    
    # Używamy stream_with_context dla długotrwałego połączenia
    @stream_with_context
    def generate_stream():
        if not features:
            yield json.dumps({"error": "Nie wybrano cech"}) + "\n"
            return
    
        try:
            df = prepare_data_with_features(ticker)
            
            # Target: Cena za 5 dni
            shift_days = -5
            df['Future_Return'] = df['Price'].shift(shift_days) / df['Price'] - 1
            df['Target_Scaled'] = df['Future_Return'] * 100 
            
            # Podział danych
            df_train = df.dropna(subset=['Target_Scaled']).copy()
            df_future = df[df['Target_Scaled'].isna()].copy()
            
            if df_future.empty:
                df_future = df.tail(shift_days * -1).copy()

            # Wywołanie generatora
            yield from neuro_engine.train_model(
                df_train,
                df_future,
                features=features, 
                target_col='Target_Scaled', 
                epochs=epochs,
                lr=0.01
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"error": str(e)}) + "\n"

    # Zwracamy odpowiedź strumieniową
    return Response(generate_stream(), mimetype='application/x-json-stream')

@app.route('/api/optimize', methods=['POST'])
def optimize():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    @stream_with_context
    def generate():
        try:
            df = data_engine.prepare_dataset(ticker)
            def on_progress(percent):
                yield json.dumps({"status": "progress", "value": percent}) + "\n"
            best_weights = optimizer.optimize_weights(df, progress_callback=on_progress)
            yield json.dumps({"status": "done", "result": best_weights}) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
    return Response(generate(), mimetype='application/x-json-stream')

if __name__ == '__main__':
    app.run(debug=True)