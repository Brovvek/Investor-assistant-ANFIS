import pandas as pd
import numpy as np

class BacktestEngine:
    def run(self, df, initial_capital=10000, 
            buy_threshold=70, sell_threshold=30, 
            stop_loss_pct=0.05, 
            take_profit_pct=0.15,
            use_trailing_stop=False,
            trailing_stop_pct=0.05,
            fee_pct=0.001):
        
        if df.empty: return {}

        cash = initial_capital
        position = 0 
        trades = []
        
        # Historie do wykresu
        equity_curve = []      # Nasza strategia
        bnh_equity_curve = []  # Buy & Hold (Benchmark)
        dates = []

        # Benchmark setup
        initial_price = df.iloc[0]['Price']
        # Zakładamy, że w B&H też płacimy prowizję na starcie
        bnh_shares = (initial_capital * (1 - fee_pct)) / initial_price
        
        in_market = False
        entry_price = 0.0
        highest_price_since_entry = 0.0

        for index, row in df.iterrows():
            price = row['Price']
            score = row['Sentiment_Oscillator']
            date = row['Date'] # String daty
            
            action = None
            reason = ""

            # --- LOGIKA WYJŚCIA ---
            if in_market:
                highest_price_since_entry = max(highest_price_since_entry, price)
                
                # Warunki sprzedaży
                if price <= entry_price * (1 - stop_loss_pct):
                    action = "SELL"; reason = "Stop Loss"
                elif use_trailing_stop and price <= highest_price_since_entry * (1 - trailing_stop_pct):
                    action = "SELL"; reason = "Trailing Stop"
                elif take_profit_pct > 0 and price >= entry_price * (1 + take_profit_pct):
                    action = "SELL"; reason = "Take Profit"
                elif score <= sell_threshold:
                    action = "SELL"; reason = "ANFIS Signal"

            # --- LOGIKA WEJŚCIA ---
            elif not in_market:
                if score >= buy_threshold:
                    action = "BUY"; reason = "ANFIS Signal"

            # --- WYKONANIE TRANSAKCJI ---
            if action == "BUY":
                cost = price * (1 + fee_pct)
                if cash >= cost:
                    position = cash / cost
                    cash = 0
                    in_market = True
                    entry_price = price
                    highest_price_since_entry = price
                    trades.append({
                        "date": date, "type": "BUY", "price": price, 
                        "score": score, "value": position * price,
                        "reason": reason
                    })

            elif action == "SELL":
                proceeds = (position * price) * (1 - fee_pct)
                profit_pct = ((proceeds / (position * entry_price * (1 + fee_pct))) - 1) * 100
                cash = proceeds
                position = 0
                in_market = False
                trades.append({
                    "date": date, "type": "SELL", "price": price, 
                    "score": score, "value": cash, 
                    "profit_pct": profit_pct,
                    "reason": reason
                })

            # --- AKTUALIZACJA STANU PORTFELA ---
            # 1. Nasz portfel
            current_val = cash if not in_market else (position * price)
            equity_curve.append(current_val)
            
            # 2. Benchmark (Buy & Hold)
            bnh_val = bnh_shares * price
            bnh_equity_curve.append(bnh_val)
            
            dates.append(date)

        # --- FINALNE METRYKI ---
        final_value = equity_curve[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        bnh_final = bnh_equity_curve[-1]
        bnh_return = ((bnh_final - initial_capital) / initial_capital) * 100

        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Win Rate
        winning_trades = len([t for t in trades if t['type'] == 'SELL' and t['profit_pct'] > 0])
        total_sell_trades = len([t for t in trades if t['type'] == 'SELL'])
        win_rate = (winning_trades / total_sell_trades * 100) if total_sell_trades > 0 else 0

        # Sharpe Ratio
        daily_returns = equity_series.pct_change().dropna()
        sharpe_ratio = 0.0
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        return {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return, 2),
            "buy_and_hold_return": round(bnh_return, 2),
            "trades_count": len(trades),
            "max_drawdown": round(max_drawdown, 2),
            "win_rate": round(win_rate, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "trades": trades[-20:], # Ostatnie 20
            # DANE DO WYKRESU:
            "chart_dates": dates,
            "equity_curve": equity_curve,
            "bnh_curve": bnh_equity_curve
        }