#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS Machine Learning Engine for Price Prediction
    Integrates PyTorch-based ANFIS with the Investor Assistant
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Import ANFIS components
from anfis_torch import make_anfis, AnfisNet, GaussMembFunc, BellMembFunc, TriangularMembFunc

dtype = torch.float


class AnfisMLEngine:
    """
    Engine for training ANFIS models to predict asset prices.
    """
    
    def __init__(self):
        self.model = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.training_history = []
        self.feature_names = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, df, feature_config, target_col='Price', lookahead=1):
        """
        Prepare data for ANFIS training.
        
        Args:
            df: DataFrame with features
            feature_config: dict of enabled features with their settings
            target_col: column to predict
            lookahead: how many days ahead to predict
            
        Returns:
            X_train, X_test, y_train, y_test, dates_test, feature_names
        """
        # Get active features
        active_features = [k for k, v in feature_config.items() 
                          if v.get('enabled', False) and k in df.columns]
        
        if not active_features:
            raise ValueError("No active features found in data")
        
        self.feature_names = active_features
        
        # Create target: price N days ahead
        df = df.copy()
        df['Target'] = df[target_col].shift(-lookahead)
        
        # Drop NaN rows
        df.dropna(subset=active_features + ['Target'], inplace=True)
        
        if len(df) < 100:
            raise ValueError(f"Not enough data: {len(df)} rows")
        
        # Extract features and target
        X = df[active_features].values
        y = df['Target'].values.reshape(-1, 1)
        dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Date'])
        
        # Scale data to [0, 1] range (important for ANFIS)
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Train/test split (80/20)
        split_idx = int(len(X_scaled) * 0.8)
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y_scaled[:split_idx]
        y_test = y_scaled[split_idx:]
        dates_test = dates[split_idx:]
        
        return X_train, X_test, y_train, y_test, dates_test
    
    def create_data_loader(self, X, y, batch_size=64):
        """Create PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.tensor(X, dtype=dtype)
        y_tensor = torch.tensor(y, dtype=dtype)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def build_model(self, X_train, num_mfs=3, hybrid=True, mf_type='gauss'):
        """
        Build ANFIS model.
        
        Args:
            X_train: training data to determine input ranges
            num_mfs: number of membership functions per variable
            hybrid: use hybrid learning (LSE + backprop)
            mf_type: 'gauss', 'bell', or 'tri'
        """
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
              optimizer_type='adam', progress_callback=None):
        """
        Train the ANFIS model.
        
        Args:
            X_train, y_train: training data
            X_test, y_test: validation data
            epochs: number of training epochs
            batch_size: mini-batch size
            learning_rate: learning rate for optimizer
            optimizer_type: 'adam', 'sgd', or 'rprop'
            progress_callback: function(epoch, total, metrics) called after each epoch
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        # Create data loaders
        train_loader = self.create_data_loader(X_train, y_train, batch_size)
        
        # Setup optimizer
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'rprop':
            optimizer = torch.optim.Rprop(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        criterion = torch.nn.MSELoss()
        
        # Training history
        history = {
            'epoch': [],
            'train_loss': [],
            'train_rmse': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mape': []
        }
        
        # Convert test data to tensors
        X_test_tensor = torch.tensor(X_test, dtype=dtype).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=dtype).to(self.device)
        
        print(f"Training ANFIS on {self.device}")
        print(f"Features: {self.feature_names}")
        print(f"Rules: {self.model.num_rules}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Hybrid learning: fit coefficients after each epoch
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
                
                # Calculate metrics
                train_loss = epoch_loss / num_batches
                train_rmse = np.sqrt(train_loss)
                val_rmse = np.sqrt(val_loss)
                
                # MAPE on original scale
                y_val_pred_orig = self.scaler_y.inverse_transform(
                    y_val_pred.cpu().numpy()
                )
                y_test_orig = self.scaler_y.inverse_transform(y_test)
                mape = np.mean(np.abs((y_test_orig - y_val_pred_orig) / 
                                      (y_test_orig + 1e-9))) * 100
            
            # Record history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_rmse'].append(train_rmse)
            history['val_loss'].append(val_loss)
            history['val_rmse'].append(val_rmse)
            history['val_mape'].append(mape)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, epochs, {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_mape': mape
                })
            
            # Print progress
            if epochs <= 30 or (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{epochs}: "
                      f"Train Loss={train_loss:.6f}, "
                      f"Val Loss={val_loss:.6f}, "
                      f"Val RMSE={val_rmse:.6f}, "
                      f"MAPE={mape:.2f}%")
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.training_history = history
        return history
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=dtype).to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
    
    def get_membership_functions(self):
        """
        Extract membership function parameters for visualization.
        
        Returns:
            dict with MF parameters for each input variable
        """
        if self.model is None:
            return {}
        
        mf_data = {}
        
        for var_name, fv in self.model.input_variables():
            mf_data[var_name] = {
                'mfs': [],
                'x_range': None
            }
            
            for mf_name, mf_obj in fv.members():
                mf_info = {
                    'name': mf_name,
                    'type': mf_obj.__class__.__name__,
                    'params': {}
                }
                
                for param_name, param in mf_obj.named_parameters():
                    mf_info['params'][param_name] = float(param.item())
                
                mf_data[var_name]['mfs'].append(mf_info)
            
            # Calculate x range for this variable
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
        """
        Evaluate membership functions for given x values.
        
        Args:
            var_name: name of the input variable
            x_values: array of x values to evaluate
            
        Returns:
            dict with membership values for each MF
        """
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
        """Get human-readable description of the fuzzy rules."""
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
                    'weight': 1.0  # All rules have equal weight in basic ANFIS
                })
        
        return rules
    
    def get_model_summary(self):
        """Get summary of the trained model."""
        if self.model is None:
            return {}
        
        return {
            'description': self.model.description,
            'num_inputs': self.model.num_in,
            'num_rules': self.model.num_rules,
            'num_outputs': self.model.num_out,
            'hybrid_learning': self.model.hybrid,
            'feature_names': self.feature_names,
            'device': str(self.device)
        }
    
    def full_training_pipeline(self, df, feature_config, 
                               num_mfs=3, epochs=100, batch_size=64,
                               learning_rate=0.01, mf_type='gauss',
                               hybrid=True, optimizer_type='adam',
                               lookahead=1, progress_callback=None):
        """
        Complete training pipeline.
        
        Returns comprehensive results for frontend visualization.
        """
        try:
            # Prepare data
            X_train, X_test, y_train, y_test, dates_test = self.prepare_data(
                df, feature_config, lookahead=lookahead
            )
            
            print(f"Data prepared: Train={len(X_train)}, Test={len(X_test)}")
            print(f"Features: {self.feature_names}")
            
            # Build model
            self.build_model(X_train, num_mfs=num_mfs, hybrid=hybrid, mf_type=mf_type)
            
            # Train model
            history = self.train(
                X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                progress_callback=progress_callback
            )
            
            # Get predictions
            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)
            y_test_actual = self.scaler_y.inverse_transform(y_test)
            y_train_actual = self.scaler_y.inverse_transform(y_train)
            
            # Calculate final metrics
            test_mse = np.mean((y_test_actual - y_test_pred) ** 2)
            test_rmse = np.sqrt(test_mse)
            test_mae = np.mean(np.abs(y_test_actual - y_test_pred))
            test_mape = np.mean(np.abs((y_test_actual - y_test_pred) / 
                                       (y_test_actual + 1e-9))) * 100
            
            # Direction accuracy
            if len(y_test_actual) > 1:
                actual_direction = np.sign(np.diff(y_test_actual.flatten()))
                pred_direction = np.sign(np.diff(y_test_pred.flatten()))
                direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            else:
                direction_accuracy = 0
            
            # Get membership functions data
            mf_data = self.get_membership_functions()
            
            # Prepare MF visualization data
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
            
            # Format dates for JSON
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
                    'val_mape': history['val_mape']
                },
                'metrics': {
                    'test_mse': float(test_mse),
                    'test_rmse': float(test_rmse),
                    'test_mae': float(test_mae),
                    'test_mape': float(test_mape),
                    'direction_accuracy': float(direction_accuracy),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'predictions': {
                    'dates': dates_str,
                    'actual': y_test_actual.flatten().tolist(),
                    'predicted': y_test_pred.flatten().tolist()
                },
                'membership_functions': mf_plots,
                'rules': self.get_rules_description()[:20],  # Limit rules for display
                'feature_importance': self._calculate_feature_importance(X_test, y_test)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _calculate_feature_importance(self, X_test, y_test):
        """Calculate feature importance using permutation method."""
        if self.model is None or len(self.feature_names) == 0:
            return {}
        
        try:
            # Baseline predictions
            base_pred = self.predict(X_test)
            base_mse = np.mean((self.scaler_y.inverse_transform(y_test) - base_pred) ** 2)
            
            importance = {}
            for i, feat_name in enumerate(self.feature_names):
                # Permute feature
                X_permuted = X_test.copy()
                np.random.shuffle(X_permuted[:, i])
                
                # Predict with permuted feature
                perm_pred = self.predict(X_permuted)
                perm_mse = np.mean((self.scaler_y.inverse_transform(y_test) - perm_pred) ** 2)
                
                # Importance = increase in error
                importance[feat_name] = max(0, (perm_mse - base_mse) / (base_mse + 1e-9))
            
            # Normalize
            total = sum(importance.values()) + 1e-9
            importance = {k: round(v / total * 100, 2) for k, v in importance.items()}
            
            return importance
        except:
            return {}


# Singleton instance
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AnfisMLEngine()
    return _engine_instance
