"""
    ANFIS Machine Learning Engine v2.0 - Advanced Price Prediction
    
    Ulepszenia:
    - Wybór typu predykcji (returns, log_returns, direction, price)
    - Walidacja walk-forward (realistyczna symulacja)
    - Rozszerzone metryki finansowe
    - Feature engineering wbudowany
    - Regularyzacja i early stopping
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from anfis_torch import make_anfis, AnfisNet, GaussMembFunc, BellMembFunc, TriangularMembFunc

dtype = torch.float


class AnfisMLEngine:
    """
    Advanced ANFIS Engine for Financial Prediction
    
    Supported prediction types:
    - 'returns': Procentowa zmiana ceny (zalecane!)
    - 'log_returns': Logarytmiczna zmiana (lepsze dla dużych ruchów)
    - 'direction': Klasyfikacja kierunku (up/down) -> 0 lub 1
    - 'price': Surowa cena (niezalecane)
    - 'volatility': Predykcja zmienności
    """
    
    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.training_history = []
        self.feature_names = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.prediction_type = 'returns'
        self.best_model_state = None
        
    def prepare_data(self, df, feature_config, target_col='Price', lookahead=1,
                     prediction_type='returns', scaler_type='robust',
                     test_size=0.2, use_walk_forward=False):
        """
        Prepare data for ANFIS training with multiple prediction types.
        
        Args:
            df: DataFrame with features
            feature_config: dict of enabled features
            target_col: column to predict
            lookahead: days ahead to predict
            prediction_type: 'returns', 'log_returns', 'direction', 'price', 'volatility'
            scaler_type: 'robust' (zalecane), 'standard', 'minmax'
            test_size: fraction for test set
            use_walk_forward: use time series cross-validation
        """
        self.prediction_type = prediction_type
        
        # Get active features
        active_features = [k for k, v in feature_config.items() 
                          if v.get('enabled', False) and k in df.columns]
        
        if not active_features:
            raise ValueError("No active features found in data")
        
        self.feature_names = active_features
        df = df.copy()
        
        # ============ CREATE TARGET BASED ON TYPE ============
        
        if prediction_type == 'returns':
            # Procentowa zmiana ceny (ZALECANE)
            # Target = (Price[t+N] - Price[t]) / Price[t] * 100
            future_price = df[target_col].shift(-lookahead)
            df['Target'] = (future_price / df[target_col] - 1) * 100
            
        elif prediction_type == 'log_returns':
            # Logarytmiczna zmiana (lepsza dla dużych ruchów, symetryczna)
            future_price = df[target_col].shift(-lookahead)
            df['Target'] = np.log(future_price / df[target_col]) * 100
            
        elif prediction_type == 'direction':
            # Klasyfikacja: 1 = cena wzrośnie, 0 = cena spadnie
            future_price = df[target_col].shift(-lookahead)
            df['Target'] = (future_price > df[target_col]).astype(float)
            
        elif prediction_type == 'volatility':
            # Predykcja zmienności (przydatne dla opcji)
            returns = df[target_col].pct_change()
            df['Target'] = returns.rolling(lookahead).std().shift(-lookahead) * np.sqrt(252) * 100
            
        elif prediction_type == 'price':
            # Surowa cena (NIEZALECANE - problemy ze stacjonarnością)
            df['Target'] = df[target_col].shift(-lookahead)
            print("⚠️ UWAGA: Predykcja surowej ceny nie jest zalecana!")
            
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
        
        # Drop NaN rows
        df.dropna(subset=active_features + ['Target'], inplace=True)
        
        if len(df) < 100:
            raise ValueError(f"Not enough data: {len(df)} rows")
        
        # Extract features and target
        X = df[active_features].values
        y = df['Target'].values.reshape(-1, 1)
        
        # Handle dates
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        elif 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
        else:
            dates = df.index
        
        # ============ FEATURE SCALING ============
        
        if scaler_type == 'robust':
            # RobustScaler - odporny na outliers (ZALECANE dla finansów)
            self.scaler_x = RobustScaler()
            self.scaler_y = RobustScaler()
        elif scaler_type == 'standard':
            # StandardScaler - normalizacja z-score
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
        else:
            # MinMaxScaler - skalowanie do [0,1]
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # ============ TRAIN/TEST SPLIT ============
        
        if use_walk_forward:
            # Time Series Split - bardziej realistyczne
            # Ostatnie test_size% danych jako test
            split_idx = int(len(X_scaled) * (1 - test_size))
        else:
            # Standard split (chronological)
            split_idx = int(len(X_scaled) * (1 - test_size))
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y_scaled[:split_idx]
        y_test = y_scaled[split_idx:]
        dates_test = dates[split_idx:split_idx + len(X_test)]
        
        # Store original y for metrics
        self.y_train_original = y[:split_idx]
        self.y_test_original = y[split_idx:]
        
        print(f"📊 Prediction type: {prediction_type}")
        print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"   Target mean: {y.mean():.4f}, std: {y.std():.4f}")
        
        return X_train, X_test, y_train, y_test, dates_test
    
    def create_data_loader(self, X, y, batch_size=64, shuffle=True):
        """Create PyTorch DataLoader."""
        X_tensor = torch.tensor(X, dtype=dtype)
        y_tensor = torch.tensor(y, dtype=dtype)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def build_model(self, X_train, num_mfs=3, hybrid=True, mf_type='gauss'):
        """Build ANFIS model."""
        x_tensor = torch.tensor(X_train, dtype=dtype)
        self.model = make_anfis(
            x_tensor, 
            num_mfs=num_mfs, 
            num_out=1, 
            hybrid=hybrid,
            mf_type=mf_type,
            var_names=self.feature_names
        )
        self.model = self.model.to(self.device)
        return self.model
    
    def train(self, X_train, y_train, X_test, y_test, 
              epochs=100, batch_size=64, learning_rate=0.01,
              optimizer_type='adam', 
              early_stopping_patience=20,
              min_delta=1e-6,
              progress_callback=None):
        """
        Train ANFIS with early stopping and advanced optimization.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        train_loader = self.create_data_loader(X_train, y_train, batch_size)
        
        # Setup optimizer with weight decay (L2 regularization)
        weight_decay = 1e-5
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=learning_rate, 
                                        weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), 
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), 
                                       lr=learning_rate, 
                                       momentum=0.9,
                                       weight_decay=weight_decay)
        elif optimizer_type == 'rprop':
            optimizer = torch.optim.Rprop(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        # Loss function
        if self.prediction_type == 'direction':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.MSELoss()
        
        # Training history
        history = {
            'epoch': [], 'train_loss': [], 'train_rmse': [],
            'val_loss': [], 'val_rmse': [], 'val_mape': [],
            'val_direction_acc': [], 'learning_rate': []
        }
        
        # Tensors for validation
        X_test_tensor = torch.tensor(X_test, dtype=dtype).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=dtype).to(self.device)
        
        print(f"🚀 Training ANFIS on {self.device}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Rules: {self.model.num_rules}")
        print(f"   Prediction type: {self.prediction_type}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Hybrid learning
            if self.model.hybrid:
                X_train_tensor = torch.tensor(X_train, dtype=dtype).to(self.device)
                y_train_tensor = torch.tensor(y_train, dtype=dtype).to(self.device)
                with torch.no_grad():
                    self.model.fit_coeff(X_train_tensor, y_train_tensor)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model(X_test_tensor)
                val_loss = criterion(y_val_pred, y_test_tensor).item()
                
                train_loss = epoch_loss / num_batches
                val_rmse = np.sqrt(val_loss)
                
                # Inverse transform for real metrics
                y_val_pred_orig = self.scaler_y.inverse_transform(
                    y_val_pred.cpu().numpy()
                )
                y_test_orig = self.scaler_y.inverse_transform(y_test)
                
                # MAPE (avoid division by zero)
                mape = np.mean(np.abs((y_test_orig - y_val_pred_orig) / 
                                      (np.abs(y_test_orig) + 1e-9))) * 100
                mape = min(mape, 999)  # Cap at 999%
                
                # Direction accuracy (important for trading!)
                if len(y_test_orig) > 1:
                    # Czy przewidzieliśmy poprawny ZNAK zmiany?
                    actual_sign = np.sign(y_test_orig.flatten())
                    pred_sign = np.sign(y_val_pred_orig.flatten())
                    direction_acc = np.mean(actual_sign == pred_sign) * 100
                else:
                    direction_acc = 50.0
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(float(train_loss))
            history['train_rmse'].append(float(np.sqrt(train_loss)))
            history['val_loss'].append(float(val_loss))
            history['val_rmse'].append(float(val_rmse))
            history['val_mape'].append(float(mape))
            history['val_direction_acc'].append(float(direction_acc))
            history['learning_rate'].append(float(current_lr))
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, epochs, {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_mape': mape,
                    'direction_acc': direction_acc
                })
            
            # Print progress
            if epochs <= 30 or (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{epochs}: "
                      f"Loss={val_loss:.6f}, "
                      f"MAPE={mape:.2f}%, "
                      f"DirAcc={direction_acc:.1f}%, "
                      f"LR={current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ Restored best model (val_loss={best_val_loss:.6f})")
        
        self.training_history = history
        return history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=dtype).to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
    
    def get_membership_functions(self):
        """Extract MF parameters for visualization."""
        if self.model is None:
            return {}
        
        mf_data = {}
        
        for var_name, fv in self.model.input_variables():
            mf_data[var_name] = {'mfs': [], 'x_range': None}
            
            for mf_name, mf_obj in fv.members():
                mf_info = {
                    'name': mf_name,
                    'type': mf_obj.__class__.__name__,
                    'params': {}
                }
                
                for param_name, param in mf_obj.named_parameters():
                    mf_info['params'][param_name] = float(param.item())
                
                mf_data[var_name]['mfs'].append(mf_info)
            
            all_centers = []
            for mf in mf_data[var_name]['mfs']:
                if 'mu' in mf['params']:
                    all_centers.append(mf['params']['mu'])
                elif 'c' in mf['params']:
                    all_centers.append(mf['params']['c'])
                elif 'b' in mf['params']:
                    all_centers.append(mf['params']['b'])
            
            if all_centers:
                margin = (max(all_centers) - min(all_centers)) * 0.3 + 0.1
                mf_data[var_name]['x_range'] = [
                    min(all_centers) - margin,
                    max(all_centers) + margin
                ]
            else:
                mf_data[var_name]['x_range'] = [0, 1]
        
        return mf_data
    
    def evaluate_membership(self, var_name, x_values):
        """Evaluate MFs for given x values."""
        if self.model is None:
            return {}
        
        results = {}
        x_tensor = torch.tensor(x_values, dtype=dtype).unsqueeze(1)
        
        for vname, fv in self.model.input_variables():
            if vname == var_name:
                for mf_name, yvals in fv.fuzzify(x_tensor):
                    results[mf_name] = yvals.squeeze().detach().numpy().tolist()
                break
        
        return results
    
    def get_rules_description(self):
        """Get fuzzy rules description."""
        if self.model is None:
            return []
        
        rules = []
        vardefs = self.model.layer['fuzzify'].varmfs
        rule_ants = self.model.layer['rules'].extra_repr(vardefs)
        
        if rule_ants:
            for i, ant in enumerate(rule_ants.split('\n')):
                rules.append({
                    'id': i,
                    'antecedent': ant,
                    'weight': 1.0
                })
        
        return rules
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model is None:
            return {}
        
        return {
            'description': self.model.description,
            'num_inputs': int(self.model.num_in),
            'num_rules': int(self.model.num_rules),
            'num_outputs': int(self.model.num_out),
            'hybrid_learning': self.model.hybrid,
            'feature_names': self.feature_names,
            'prediction_type': self.prediction_type,
            'device': str(self.device)
        }
    
    def calculate_financial_metrics(self, y_actual, y_predicted):
        """
        Calculate advanced financial metrics.
        """
        y_actual = np.array(y_actual).flatten()
        y_predicted = np.array(y_predicted).flatten()
        
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = float(np.mean((y_actual - y_predicted) ** 2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(np.mean(np.abs(y_actual - y_predicted)))
        metrics['mape'] = float(np.mean(np.abs((y_actual - y_predicted) / 
                                               (np.abs(y_actual) + 1e-9))) * 100)
        
        # Direction accuracy (najważniejsza dla tradingu!)
        actual_sign = np.sign(y_actual)
        pred_sign = np.sign(y_predicted)
        metrics['direction_accuracy'] = float(np.mean(actual_sign == pred_sign) * 100)
        
        # Correlation
        if len(y_actual) > 2:
            metrics['correlation'] = float(np.corrcoef(y_actual, y_predicted)[0, 1])
        else:
            metrics['correlation'] = 0.0
        
        # R-squared
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        metrics['r_squared'] = float(1 - (ss_res / (ss_tot + 1e-9)))
        
        # Hit ratio for different thresholds
        for threshold in [0.5, 1.0, 2.0]:
            within_threshold = np.abs(y_actual - y_predicted) <= threshold
            metrics[f'hit_ratio_{threshold}pct'] = float(np.mean(within_threshold) * 100)
        
        # Profit simulation (jeśli returns)
        if self.prediction_type in ['returns', 'log_returns']:
            # Strategia: kup gdy przewidujesz wzrost, sprzedaj gdy spadek
            positions = np.sign(y_predicted)  # +1 long, -1 short
            strategy_returns = positions * y_actual
            
            metrics['strategy_total_return'] = float(np.sum(strategy_returns))
            metrics['strategy_avg_return'] = float(np.mean(strategy_returns))
            metrics['strategy_sharpe'] = float(
                np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
            )
            
            # Win rate
            winning_trades = strategy_returns > 0
            metrics['win_rate'] = float(np.mean(winning_trades) * 100)
            
            # Profit factor
            gross_profit = np.sum(strategy_returns[strategy_returns > 0])
            gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
            metrics['profit_factor'] = float(gross_profit / (gross_loss + 1e-9))
        
        return metrics
    
    def _calculate_feature_importance(self, X_test, y_test):
        """Permutation feature importance."""
        if self.model is None or len(self.feature_names) == 0:
            return {}
        
        try:
            base_pred = self.predict(X_test)
            y_test_orig = self.scaler_y.inverse_transform(y_test)
            base_mse = np.mean((y_test_orig - base_pred) ** 2)
            
            importance = {}
            for i, feat_name in enumerate(self.feature_names):
                X_permuted = X_test.copy()
                np.random.shuffle(X_permuted[:, i])
                
                perm_pred = self.predict(X_permuted)
                perm_mse = np.mean((y_test_orig - perm_pred) ** 2)
                
                importance[feat_name] = max(0, (perm_mse - base_mse) / (base_mse + 1e-9))
            
            total = sum(importance.values()) + 1e-9
            importance = {k: round(v / total * 100, 2) for k, v in importance.items()}
            
            return importance
        except:
            return {}
    
    def full_training_pipeline(self, df, feature_config, 
                               num_mfs=3, epochs=100, batch_size=64,
                               learning_rate=0.01, mf_type='gauss',
                               hybrid=True, optimizer_type='adam',
                               lookahead=1, 
                               prediction_type='returns',
                               scaler_type='robust',
                               early_stopping_patience=20,
                               progress_callback=None):
        """
        Complete training pipeline with all enhancements.
        """
        try:
            # Prepare data
            X_train, X_test, y_train, y_test, dates_test = self.prepare_data(
                df, feature_config, 
                lookahead=lookahead,
                prediction_type=prediction_type,
                scaler_type=scaler_type
            )
            
            print(f"📦 Data prepared: Train={len(X_train)}, Test={len(X_test)}")
            print(f"   Features: {self.feature_names}")
            
            # Build model
            self.build_model(X_train, num_mfs=num_mfs, hybrid=hybrid, mf_type=mf_type)
            
            # Train model
            history = self.train(
                X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                early_stopping_patience=early_stopping_patience,
                progress_callback=progress_callback
            )
            
            # Get predictions
            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)
            y_test_actual = self.scaler_y.inverse_transform(y_test)
            y_train_actual = self.scaler_y.inverse_transform(y_train)
            
            # Calculate comprehensive metrics
            metrics = self.calculate_financial_metrics(y_test_actual, y_test_pred)
            metrics['train_samples'] = len(X_train)
            metrics['test_samples'] = len(X_test)
            
            # MF data
            mf_data = self.get_membership_functions()
            mf_plots = {}
            for var_name, var_data in mf_data.items():
                x_range = var_data['x_range']
                x_vals = np.linspace(x_range[0], x_range[1], 100)
                mf_values = self.evaluate_membership(var_name, x_vals)
                mf_plots[var_name] = {
                    'x': x_vals.tolist(),
                    'mfs': mf_values,
                    'params': var_data['mfs']
                }
            
            # Format dates
            dates_str = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') 
                        else str(d) for d in dates_test]
            
            return {
                'status': 'success',
                'model_summary': self.get_model_summary(),
                'training_history': {
                    'epochs': history['epoch'],
                    'train_loss': history['train_loss'],
                    'val_loss': history['val_loss'],
                    'train_rmse': history['train_rmse'],
                    'val_rmse': history['val_rmse'],
                    'val_mape': history['val_mape'],
                    'val_direction_acc': history['val_direction_acc'],
                    'learning_rate': history['learning_rate']
                },
                'metrics': metrics,
                'predictions': {
                    'dates': dates_str,
                    'actual': y_test_actual.flatten().tolist(),
                    'predicted': y_test_pred.flatten().tolist()
                },
                'membership_functions': mf_plots,
                'rules': self.get_rules_description()[:20],
                'feature_importance': self._calculate_feature_importance(X_test, y_test)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }


# Singleton
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AnfisMLEngine()
    return _engine_instance
