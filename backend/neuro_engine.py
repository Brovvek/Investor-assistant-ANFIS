import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from anfis_pytorch.membership import make_anfis

class NeuroEngine:
    def __init__(self):
        self.model = None
        self.trained_features = []
        
    def train_model(self, df_train, df_future, features, target_col='Target', epochs=100, lr=0.01):
        """
        Generator (yield) zwracający postęp treningu oraz bogate dane wynikowe (Cena, Data).
        """
        # 1. WALIDACJA CECH
        valid_features = [f for f in features if f in df_train.columns]
        if not valid_features:
            yield json.dumps({"error": f"Brak cech {features} w danych"}) + "\n"
            return
            
        self.trained_features = valid_features

        # 2. PRZYGOTOWANIE DANYCH (Tensory)
        X_train = torch.tensor(df_train[valid_features].values, dtype=torch.float)
        y_train = torch.tensor(df_train[target_col].values, dtype=torch.float)
        X_future = torch.tensor(df_future[valid_features].values, dtype=torch.float)

        # --- NOWOŚĆ: Wyciągamy Metadane (Daty i Ceny) ---
        # Zakładamy, że Index to Data (DatetimeIndex)
        try:
            train_dates = df_train.index.strftime('%Y-%m-%d').tolist()
            future_dates = df_future.index.strftime('%Y-%m-%d').tolist()
        except:
            # Fallback jeśli index nie jest datą
            train_dates = [str(i) for i in range(len(df_train))]
            future_dates = [str(i) for i in range(len(df_train), len(df_train)+len(df_future))]

        # Wyciągamy ceny bazowe (do wyświetlania na wykresie)
        train_prices = df_train['Price'].values.tolist() if 'Price' in df_train else []
        future_base_prices = df_future['Price'].values.tolist() if 'Price' in df_future else []

        # 3. BUDOWA MODELU
        self.model = make_anfis(X_train, num_mfs=3, num_out=1)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        loss_history = []
        report_interval = max(1, int(epochs * 0.05))

        # 4. PĘTLA UCZENIA
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
            # Strumieniowanie postępu
            if epoch % report_interval == 0 or epoch == epochs - 1:
                progress = int(((epoch + 1) / epochs) * 100)
                yield json.dumps({
                    "status": "progress", 
                    "percent": progress, 
                    "loss": loss.item()
                }) + "\n"

        # 5. WYNIKI KOŃCOWE
        self.model.eval()
        with torch.no_grad():
            history_pred_pct = self.model(X_train).numpy().flatten().tolist()
            future_pred_pct = self.model(X_future).numpy().flatten().tolist()

        # --- NOWOŚĆ: Obliczamy Implikowaną Cenę Przyszłą ---
        # Wzór: Cena_Przyszła = Cena_Aktualna * (1 + Prognozowany_Zwrot / 100)
        future_implied_prices = []
        for base_price, pred_pct in zip(future_base_prices, future_pred_pct):
            implied = base_price * (1 + pred_pct / 100.0)
            future_implied_prices.append(implied)

        final_result = {
            "loss_history": loss_history,
            
            # Dane wykresowe
            "dates_history": train_dates,
            "dates_future": future_dates,
            
            "prices_history": train_prices,
            "prices_future_implied": future_implied_prices, # To pokażemy w dymku prognozy
            
            "predictions": history_pred_pct,       
            "future_predictions": future_pred_pct, 
            "actual": df_train[target_col].values.tolist(), 
            
            "final_loss": loss_history[-1],
            "features_used": valid_features
        }

        yield json.dumps({"status": "done", "data": final_result}) + "\n"

    def predict(self, inputs_dict):
        if not self.model: return 50.0
        try:
            x_vals = [inputs_dict.get(f, 0.0) for f in self.trained_features]
            x_tensor = torch.tensor([x_vals], dtype=torch.float)
            self.model.eval()
            with torch.no_grad():
                res = self.model(x_tensor)
            return res.item()
        except:
            return 50.0