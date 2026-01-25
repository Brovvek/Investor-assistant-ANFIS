from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from data_engine import DataEngine
from fuzzy_expert_system import FuzzyExpertSystem  # Klasyczny FIS Mamdani (nie ANFIS!)
from backtest_engine import BacktestEngine
from weight_optimizer import WeightOptimizer  # Optymalizator wag (nie uczenie ANFIS!)
from feature_factory import FeatureFactory

# Try to import v2, fallback to v1
try:
    from anfis_ml_engine_v2 import AnfisMLEngine
    print("✅ Using ANFIS ML Engine v2.0 (Advanced)")
except ImportError:
    from anfis_ml_engine import AnfisMLEngine
    print("⚠️ Using ANFIS ML Engine v1.0 (Basic)")

import pandas as pd
import numpy as np
import json

try:
    from config import FRED_API_KEY
except ImportError:
    FRED_API_KEY = None

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def json_dumps(obj):
    """JSON dumps with numpy support"""
    return json.dumps(obj, cls=NumpyEncoder)

app = Flask(__name__)
CORS(app)

data_engine = DataEngine(api_key=FRED_API_KEY) 
fuzzy_expert = FuzzyExpertSystem()  # Klasyczny system rozmyty (NIE ANFIS!)
backtester = BacktestEngine()
weight_optimizer = WeightOptimizer()  # Optymalizator wag metodą DE
anfis_ml = AnfisMLEngine()  # PRAWDZIWY ANFIS - sieć neuronowa z uczeniem

# --- FUNKCJE POMOCNICZE ---

def normalize_rank(series, window=252):
    """
    Normalizacja rang z oknem 252 dni (1 rok).
    """
    if series.empty: return series
    return series.rolling(window, min_periods=1).rank(pct=True) * 100

def prepare_data_with_features(ticker):
    """
    Przygotowanie danych odporne na błędy (Robust Data Prep)
    """
    # 1. Pobranie danych (z cache)
    df_raw = data_engine.prepare_dataset(ticker)
    if df_raw.empty: return pd.DataFrame()
    
    # 2. Generowanie cech
    df = FeatureFactory.generate_features(df_raw)
    
    # 3. Normalizacja Hybrydowa
    cols_to_check = [c for c in df.columns if c not in ['Date', 'Price', 'Open', 'High', 'Low', 'Close', 'Adj Close']]
    
    window = 252 
    
    for col in cols_to_check:
        if col == 'RSI' or 'RSI_' in col and 'Rank' not in col:
            df[col] = df[col].fillna(50)
        else:
            df[col] = normalize_rank(df[col], window).fillna(50)
    
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
        
        df.fillna(method='ffill', inplace=True)
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

        fuzzy_expert.build_system(config)
        
        oscillator_values = []
        df_analysis = df.tail(1260).copy()
        active_features = [k for k, v in config.items() if v.get('enabled')]

        for index, row in df_analysis.iterrows():
            inputs = {feat: row[feat] for feat in active_features if feat in row}
            score = fuzzy_expert.compute(inputs)
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
        fuzzy_expert.build_system(config)
        
        oscillator_values = []
        df_analysis = df.tail(1260).copy()
        active_features = [k for k, v in config.items() if v.get('enabled')]
        
        print(f"Backtest: Analiza {len(df_analysis)} wierszy dla {ticker}")
        
        for index, row in df_analysis.iterrows():
            inputs = {feat: row[feat] for feat in active_features if feat in row}
            oscillator_values.append(fuzzy_expert.compute(inputs))
            
        df_analysis['Sentiment_Oscillator'] = oscillator_values
        
        max_score = max(oscillator_values) if oscillator_values else 0
        min_score = min(oscillator_values) if oscillator_values else 0
        print(f"DEBUG FIS: Min={min_score:.2f}, Max={max_score:.2f}")

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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    @stream_with_context
    def generate():
        try:
            df = data_engine.prepare_dataset(ticker)
            def on_progress(percent):
                yield json_dumps({"status": "progress", "value": percent}) + "\n"
            best_weights = weight_optimizer.optimize_weights(df, progress_callback=on_progress)
            yield json_dumps({"status": "done", "result": best_weights}) + "\n"
        except Exception as e:
            yield json_dumps({"status": "error", "message": str(e)}) + "\n"
    return Response(generate(), mimetype='application/x-json-stream')


# ============================================================================
# NEW ENDPOINTS: ANFIS Machine Learning for Price Prediction
# ============================================================================

@app.route('/api/anfis_ml/train', methods=['POST'])
def train_anfis_ml():
    """
    Train ANFIS model for price prediction.
    
    Expected JSON payload:
    {
        "ticker": "^GSPC",
        "config": { "RSI": {"enabled": true, "weight": 1.0, "direction": -1}, ... },
        "epochs": 100,
        "num_mfs": 3,
        "batch_size": 64,
        "learning_rate": 0.01,
        "mf_type": "gauss",  // gauss, bell, tri
        "hybrid": true,
        "optimizer": "adam",  // adam, sgd, rprop
        "lookahead": 1  // days ahead to predict
    }
    """
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    config = req_data.get('config', {})
    
    # Training parameters
    epochs = int(req_data.get('epochs', 100))
    num_mfs = int(req_data.get('num_mfs', 3))
    batch_size = int(req_data.get('batch_size', 64))
    learning_rate = float(req_data.get('learning_rate', 0.01))
    mf_type = req_data.get('mf_type', 'gauss')
    hybrid = req_data.get('hybrid', True)
    optimizer_type = req_data.get('optimizer', 'adam')
    lookahead = int(req_data.get('lookahead', 1))
    
    try:
        # Prepare data with features
        df = prepare_data_with_features(ticker)
        if df.empty:
            return jsonify({"status": "error", "message": "Brak danych dla tego tickera"}), 400
        
        # Create new engine instance for this training
        ml_engine = AnfisMLEngine()
        
        # Run full training pipeline
        result = ml_engine.full_training_pipeline(
            df=df,
            feature_config=config,
            num_mfs=num_mfs,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mf_type=mf_type,
            hybrid=hybrid,
            optimizer_type=optimizer_type,
            lookahead=lookahead
        )
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500


@app.route('/api/anfis_ml/train_stream', methods=['POST'])
def train_anfis_ml_stream():
    """
    Train ANFIS model with streaming progress updates.
    Returns JSON lines with progress and final results.
    """
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    config = req_data.get('config', {})
    
    # Basic parameters
    epochs = int(req_data.get('epochs', 100))
    num_mfs = int(req_data.get('num_mfs', 3))
    batch_size = int(req_data.get('batch_size', 64))
    learning_rate = float(req_data.get('learning_rate', 0.01))
    mf_type = req_data.get('mf_type', 'gauss')
    hybrid = req_data.get('hybrid', True)
    optimizer_type = req_data.get('optimizer', 'adam')
    lookahead = int(req_data.get('lookahead', 1))
    
    # NEW: Advanced parameters
    prediction_type = req_data.get('prediction_type', 'returns')
    scaler_type = req_data.get('scaler_type', 'robust')
    early_stopping_patience = int(req_data.get('early_stopping_patience', 20))
    training_days = int(req_data.get('training_days', 0))  # 0 = all data
    
    @stream_with_context
    def generate():
        try:
            # Prepare data
            yield json_dumps({"status": "preparing", "message": "Przygotowywanie danych..."}) + "\n"
            
            df = prepare_data_with_features(ticker)
            if df.empty:
                yield json_dumps({"status": "error", "message": "Brak danych"}) + "\n"
                return
            
            # Filter to last N days if specified
            if training_days > 0 and len(df) > training_days:
                df = df.tail(training_days).copy()
                print(f"📅 Ograniczono dane do ostatnich {training_days} dni ({len(df)} wierszy)")
            
            # Create ML engine
            ml_engine = AnfisMLEngine()
            
            # Prepare data for training
            X_train, X_test, y_train, y_test, dates_test = ml_engine.prepare_data(
                df, config, lookahead=lookahead,
                prediction_type=prediction_type,
                scaler_type=scaler_type
            )
            
            yield json_dumps({
                "status": "data_ready",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": ml_engine.feature_names,
                "total_days": len(df),
                "training_days_requested": training_days
            }) + "\n"
            
            # Build model
            ml_engine.build_model(X_train, num_mfs=num_mfs, hybrid=hybrid, mf_type=mf_type)
            
            yield json_dumps({
                "status": "model_built",
                "num_rules": int(ml_engine.model.num_rules),
                "num_inputs": int(ml_engine.model.num_in)
            }) + "\n"
            
            # Training with progress callback
            def progress_cb(epoch, total, metrics):
                # Yield progress every 5 epochs or at start/end
                if epoch <= 5 or epoch % 5 == 0 or epoch == total:
                    pass  # Will be handled in main loop
            
            # Custom training loop with streaming
            from torch.utils.data import TensorDataset, DataLoader
            import torch
            
            train_loader = ml_engine.create_data_loader(X_train, y_train, batch_size)
            
            if optimizer_type == 'adam':
                optimizer = torch.optim.Adam(ml_engine.model.parameters(), lr=learning_rate)
            elif optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(ml_engine.model.parameters(), lr=learning_rate, momentum=0.9)
            else:
                optimizer = torch.optim.Rprop(ml_engine.model.parameters(), lr=learning_rate)
            
            criterion = torch.nn.MSELoss()
            
            X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(ml_engine.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(ml_engine.device)
            
            history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mape': [], 'val_direction_acc': [], 'learning_rate': []}
            
            for epoch in range(epochs):
                ml_engine.model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(ml_engine.device)
                    y_batch = y_batch.to(ml_engine.device)
                    
                    y_pred = ml_engine.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                # Hybrid learning
                if ml_engine.model.hybrid:
                    X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(ml_engine.device)
                    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(ml_engine.device)
                    with torch.no_grad():
                        ml_engine.model.fit_coeff(X_train_tensor, y_train_tensor)
                
                # Validation
                ml_engine.model.eval()
                with torch.no_grad():
                    y_val_pred = ml_engine.model(X_test_tensor)
                    val_loss = criterion(y_val_pred, y_test_tensor).item()
                    
                    train_loss = epoch_loss / num_batches
                    val_rmse = np.sqrt(val_loss)
                    
                    y_val_pred_orig = ml_engine.scaler_y.inverse_transform(y_val_pred.cpu().numpy())
                    y_test_orig = ml_engine.scaler_y.inverse_transform(y_test)
                    mape = np.mean(np.abs((y_test_orig - y_val_pred_orig) / (np.abs(y_test_orig) + 1e-9))) * 100
                    mape = min(mape, 999)  # Cap MAPE
                    
                    # Direction accuracy (for returns: check sign match)
                    actual_sign = np.sign(y_test_orig.flatten())
                    pred_sign = np.sign(y_val_pred_orig.flatten())
                    dir_acc = np.mean(actual_sign == pred_sign) * 100
                
                history['epoch'].append(epoch + 1)
                history['train_loss'].append(float(train_loss))
                history['val_loss'].append(float(val_loss))
                history['val_rmse'].append(float(val_rmse))
                history['val_mape'].append(float(mape))
                history['val_direction_acc'].append(float(dir_acc))
                history['learning_rate'].append(float(learning_rate))
                
                # Stream progress
                if epoch == 0 or (epoch + 1) % max(1, epochs // 20) == 0 or epoch == epochs - 1:
                    yield json_dumps({
                        "status": "training",
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "progress": int((epoch + 1) / epochs * 100),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "val_rmse": float(val_rmse),
                        "val_mape": float(mape),
                        "direction_acc": float(dir_acc)
                    }) + "\n"
            
            # Final predictions
            y_test_pred = ml_engine.predict(X_test)
            y_test_actual = ml_engine.scaler_y.inverse_transform(y_test)
            
            # Calculate metrics - use advanced method if available
            if hasattr(ml_engine, 'calculate_financial_metrics'):
                metrics = ml_engine.calculate_financial_metrics(y_test_actual, y_test_pred)
                metrics['train_samples'] = len(X_train)
                metrics['test_samples'] = len(X_test)
            else:
                # Fallback to basic metrics
                test_mse = np.mean((y_test_actual - y_test_pred) ** 2)
                test_rmse = np.sqrt(test_mse)
                test_mae = np.mean(np.abs(y_test_actual - y_test_pred))
                test_mape = np.mean(np.abs((y_test_actual - y_test_pred) / (y_test_actual + 1e-9))) * 100
                
                # Direction accuracy - for returns, check sign match
                if len(y_test_actual) > 1:
                    actual_sign = np.sign(y_test_actual.flatten())
                    pred_sign = np.sign(y_test_pred.flatten())
                    dir_acc = np.mean(actual_sign == pred_sign) * 100
                else:
                    dir_acc = 50.0
                
                # Correlation
                if len(y_test_actual) > 2:
                    correlation = float(np.corrcoef(y_test_actual.flatten(), y_test_pred.flatten())[0, 1])
                else:
                    correlation = 0.0
                
                # R-squared
                ss_res = np.sum((y_test_actual - y_test_pred) ** 2)
                ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
                r_squared = float(1 - (ss_res / (ss_tot + 1e-9)))
                
                metrics = {
                    "mse": float(test_mse),
                    "rmse": float(test_rmse),
                    "mae": float(test_mae),
                    "mape": float(min(test_mape, 999)),
                    "direction_accuracy": float(dir_acc),
                    "correlation": correlation,
                    "r_squared": r_squared,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test)
                }
            
            # MF data
            mf_plots = {}
            mf_data = ml_engine.get_membership_functions()
            for var_name, var_data in mf_data.items():
                x_range = var_data['x_range']
                x_vals = np.linspace(x_range[0], x_range[1], 100)
                mf_values = ml_engine.evaluate_membership(var_name, x_vals)
                mf_plots[var_name] = {
                    'x': x_vals.tolist(),
                    'mfs': mf_values,
                    'params': var_data['mfs']
                }
            
            # Format dates
            dates_str = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates_test]
            
            # Final result
            # Add direction_acc to history if available
            if 'val_direction_acc' not in history:
                history['val_direction_acc'] = [50.0] * len(history['epoch'])
            
            yield json_dumps({
                "status": "done",
                "model_summary": ml_engine.get_model_summary(),
                "training_history": {
                    "epochs": history['epoch'],
                    "train_loss": history['train_loss'],
                    "val_loss": history['val_loss'],
                    "val_rmse": history['val_rmse'],
                    "val_mape": history['val_mape'],
                    "val_direction_acc": history.get('val_direction_acc', [50.0] * len(history['epoch'])),
                    "learning_rate": history.get('learning_rate', [learning_rate] * len(history['epoch']))
                },
                "metrics": metrics,
                "predictions": {
                    "dates": dates_str,
                    "actual": y_test_actual.flatten().tolist(),
                    "predicted": y_test_pred.flatten().tolist()
                },
                "membership_functions": mf_plots,
                "rules": ml_engine.get_rules_description()[:20],
                "feature_importance": ml_engine._calculate_feature_importance(X_test, y_test)
            }) + "\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json_dumps({"status": "error", "message": str(e)}) + "\n"
    
    return Response(generate(), mimetype='application/x-json-stream')


@app.route('/api/anfis_ml/available_features', methods=['POST'])
def get_available_features():
    """Get list of available features for a ticker."""
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    
    try:
        df = prepare_data_with_features(ticker)
        if df.empty:
            return jsonify({"error": "Brak danych"}), 400
        
        # Get all numeric columns except price-related
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']
        features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
        
        return jsonify({
            "features": features,
            "total_rows": len(df),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- HISTORIA TRENINGÓW (CSV) ---

import os
from datetime import datetime

# Ścieżka do pliku w tym samym katalogu co app.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_HISTORY_FILE = os.path.join(_SCRIPT_DIR, 'training_history.csv')
print(f"📁 Plik historii treningów: {TRAINING_HISTORY_FILE}")

@app.route('/api/anfis_ml/save_results', methods=['POST'])
def save_training_results():
    """Zapisuje wyniki treningu do pliku CSV"""
    try:
        req_data = request.json
        
        # Generuj unikalne ID
        import uuid
        record_id = str(uuid.uuid4())[:8]  # Krótkie ID (8 znaków)
        
        # Przygotowanie wiersza danych
        row = {
            'id': record_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': req_data.get('ticker', ''),
            'prediction_type': req_data.get('prediction_type', ''),
            'epochs': req_data.get('epochs', 0),
            'num_mfs': req_data.get('num_mfs', 0),
            'mf_type': req_data.get('mf_type', ''),
            'optimizer': req_data.get('optimizer', ''),
            'learning_rate': req_data.get('learning_rate', 0),
            'batch_size': req_data.get('batch_size', 0),
            'lookahead': req_data.get('lookahead', 1),
            'training_days': req_data.get('training_days', 0),
            'scaler_type': req_data.get('scaler_type', ''),
            'hybrid': req_data.get('hybrid', False),
            'features': req_data.get('features', ''),
            # Metryki
            'direction_accuracy': req_data.get('direction_accuracy', 0),
            'rmse': req_data.get('rmse', 0),
            'mape': req_data.get('mape', 0),
            'r_squared': req_data.get('r_squared', 0),
            'correlation': req_data.get('correlation', 0),
            'train_samples': req_data.get('train_samples', 0),
            'test_samples': req_data.get('test_samples', 0),
            'win_rate': req_data.get('win_rate', 0),
            'sharpe_ratio': req_data.get('sharpe_ratio', 0),
            'num_rules': req_data.get('num_rules', 0),
            'final_epoch': req_data.get('final_epoch', 0),
            'notes': req_data.get('notes', '')
        }
        
        # Sprawdź czy plik istnieje
        file_exists = os.path.exists(TRAINING_HISTORY_FILE)
        
        # Zapisz do CSV
        df_new = pd.DataFrame([row])
        
        if file_exists:
            df_new.to_csv(TRAINING_HISTORY_FILE, mode='a', header=False, index=False)
        else:
            df_new.to_csv(TRAINING_HISTORY_FILE, mode='w', header=True, index=False)
        
        print(f"✅ Zapisano wyniki treningu [ID: {record_id}]: {row['ticker']} @ {row['timestamp']}")
        
        return jsonify({
            "success": True,
            "message": "Wyniki zapisane pomyślnie",
            "id": record_id,
            "filename": TRAINING_HISTORY_FILE
        })
        
    except Exception as e:
        print(f"❌ Błąd zapisu: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/anfis_ml/get_history', methods=['GET'])
def get_training_history():
    """Pobiera historię treningów z pliku CSV"""
    try:
        print(f"📂 Szukam pliku: {TRAINING_HISTORY_FILE}")
        
        if not os.path.exists(TRAINING_HISTORY_FILE):
            print("⚠️ Plik historii nie istnieje")
            return Response(
                json.dumps({"history": [], "total_records": 0, "stats": {}, "message": "Brak pliku"}),
                mimetype='application/json'
            )
        
        # Wczytaj CSV
        df = pd.read_csv(TRAINING_HISTORY_FILE)
        print(f"📊 Wczytano {len(df)} rekordów z CSV")
        print(f"📊 Kolumny: {list(df.columns)}")
        
        if df.empty:
            return Response(
                json.dumps({"history": [], "total_records": 0, "stats": {}, "message": "Plik pusty"}),
                mimetype='application/json'
            )
        
        # Zamień NaN na puste stringi
        df = df.fillna('')
        
        # Sortuj od najnowszych
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False)
        
        # Konwertuj do listy słowników - ręcznie dla pewności
        history = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                # Konwertuj numpy types do Python types
                if pd.isna(val):
                    record[col] = ''
                elif hasattr(val, 'item'):  # numpy scalar
                    record[col] = val.item()
                else:
                    record[col] = val
            history.append(record)
        
        print(f"📊 Przygotowano {len(history)} rekordów")
        
        # Statystyki
        stats = {'total_trainings': len(history)}
        try:
            if 'direction_accuracy' in df.columns:
                da_values = pd.to_numeric(df['direction_accuracy'], errors='coerce').dropna()
                if len(da_values) > 0:
                    stats['avg_direction_accuracy'] = round(float(da_values.mean()), 2)
                    stats['max_direction_accuracy'] = round(float(da_values.max()), 2)
            if 'rmse' in df.columns:
                rmse_values = pd.to_numeric(df['rmse'], errors='coerce').dropna()
                if len(rmse_values) > 0:
                    stats['avg_rmse'] = round(float(rmse_values.mean()), 4)
        except Exception as e:
            print(f"⚠️ Błąd statystyk: {e}")
        
        result = {
            "history": history,
            "total_records": len(history),
            "stats": stats
        }
        
        print(f"✅ Zwracam odpowiedź z {len(history)} rekordami")
        
        # Użyj json.dumps zamiast jsonify dla pewności
        return Response(
            json.dumps(result, ensure_ascii=False, default=str),
            mimetype='application/json'
        )
        
    except Exception as e:
        import traceback
        print(f"❌ Błąd odczytu historii: {e}")
        traceback.print_exc()
        return Response(
            json.dumps({"error": str(e), "history": [], "total_records": 0, "stats": {}}),
            mimetype='application/json'
        ), 500


@app.route('/api/anfis_ml/delete_history', methods=['DELETE'])
def delete_training_history():
    """Usuwa wybrane rekordy lub całą historię treningów"""
    try:
        req_data = request.json or {}
        ids_to_delete = req_data.get('ids', [])
        
        if not os.path.exists(TRAINING_HISTORY_FILE):
            return jsonify({"success": True, "message": "Brak historii do usunięcia", "deleted": 0})
        
        # Jeśli nie podano ID, usuń cały plik
        if not ids_to_delete:
            os.remove(TRAINING_HISTORY_FILE)
            print("🗑️ Usunięto całą historię treningów")
            return jsonify({"success": True, "message": "Cała historia usunięta", "deleted": "all"})
        
        # Wczytaj CSV
        df = pd.read_csv(TRAINING_HISTORY_FILE)
        original_count = len(df)
        
        # Sprawdź czy kolumna 'id' istnieje
        if 'id' not in df.columns:
            return jsonify({"success": False, "error": "Brak kolumny ID w pliku historii"}), 400
        
        # Filtruj - zostaw tylko te, których ID NIE ma na liście do usunięcia
        df_filtered = df[~df['id'].isin(ids_to_delete)]
        deleted_count = original_count - len(df_filtered)
        
        # Zapisz z powrotem
        if len(df_filtered) == 0:
            os.remove(TRAINING_HISTORY_FILE)
            print(f"🗑️ Usunięto wszystkie {deleted_count} rekordy (plik usunięty)")
        else:
            df_filtered.to_csv(TRAINING_HISTORY_FILE, index=False)
            print(f"🗑️ Usunięto {deleted_count} rekordów, pozostało {len(df_filtered)}")
        
        return jsonify({
            "success": True, 
            "message": f"Usunięto {deleted_count} rekordów",
            "deleted": deleted_count,
            "remaining": len(df_filtered)
        })
        
    except Exception as e:
        print(f"❌ Błąd usuwania: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/anfis_ml/export_selected', methods=['POST'])
def export_selected_history():
    """Eksportuje wybrane rekordy jako plik CSV"""
    try:
        req_data = request.json or {}
        ids_to_export = req_data.get('ids', [])
        
        if not os.path.exists(TRAINING_HISTORY_FILE):
            return jsonify({"error": "Brak historii do eksportu"}), 404
        
        df = pd.read_csv(TRAINING_HISTORY_FILE)
        
        # Jeśli podano ID, filtruj
        if ids_to_export and 'id' in df.columns:
            df = df[df['id'].isin(ids_to_export)]
        
        if df.empty:
            return jsonify({"error": "Brak rekordów do eksportu"}), 404
        
        # Konwertuj do CSV
        csv_content = df.to_csv(index=False)
        
        print(f"📥 Eksportowano {len(df)} rekordów")
        
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=training_history_selected.csv'}
        )
        
    except Exception as e:
        print(f"❌ Błąd eksportu: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/anfis_ml/export_history', methods=['GET'])
def export_training_history():
    """Eksportuje historię jako plik CSV do pobrania"""
    try:
        if not os.path.exists(TRAINING_HISTORY_FILE):
            return jsonify({"error": "Brak historii do eksportu"}), 404
        
        with open(TRAINING_HISTORY_FILE, 'r', encoding='utf-8') as f:
            csv_content = f.read()
        
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=training_history.csv'}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
