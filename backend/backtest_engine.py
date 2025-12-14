import pandas as pd

class BacktestEngine:
    def run(self, df, initial_capital=10000, buy_threshold=70, sell_threshold=30):
        cash = initial_capital
        position = 0 
        trades = []
        equity_curve = [] 
        
        if df.empty: return {}

        buy_and_hold_shares = initial_capital / df.iloc[0]['Price']
        in_market = False
        entry_price = 0.0

        for index, row in df.iterrows():
            price = row['Price']
            score = row['Sentiment_Oscillator']
            date = row['Date']

            # KUPNO
            if score >= buy_threshold and not in_market:
                position = cash / price
                cash = 0
                in_market = True
                entry_price = price
                trades.append({
                    "date": date, "type": "BUY", "price": price, 
                    "score": score, "value": position * price
                })

            # SPRZEDAŻ
            elif score <= sell_threshold and in_market:
                cash = position * price
                position = 0
                in_market = False
                trades.append({
                    "date": date, "type": "SELL", "price": price, 
                    "score": score, "value": cash, 
                    "profit_pct": ((price - entry_price) / entry_price) * 100
                })

            current_val = cash if not in_market else (position * price)
            equity_curve.append(current_val)

        final_value = equity_curve[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        bnh_final = buy_and_hold_shares * df.iloc[-1]['Price']
        bnh_return = ((bnh_final - initial_capital) / initial_capital) * 100

        return {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return, 2),
            "buy_and_hold_return": round(bnh_return, 2),
            "trades_count": len(trades),
            "trades": trades[-10:] 
        }