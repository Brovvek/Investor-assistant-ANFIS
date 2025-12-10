from flask import Flask, render_template, jsonify, request
from data_engine import DataEngine
from anfis_engine import AnfisEngine
import pandas as pd
import json

app = Flask(__name__)

# --- KONFIGURACJA ---
# PAMIĘTAJ: Wklej tutaj swój klucz API z fred.stlouisfed.org
FRED_API_KEY = 'b7d804e08b899c4a8c9fdfff48dfdad8' 

data_engine = DataEngine(FRED_API_KEY)
anfis_engine = AnfisEngine()

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    
    try:
        # 1. Pobieranie danych (Teraz DataEngine pobiera RSI, VIX i Yield)
        df = data_engine.prepare_dataset(ticker)
        
        if df.empty:
            return jsonify({"error": "Brak danych"}), 400

        # 2. Normalizacja danych wejściowych do skali 0-100 dla ANFIS
        # Używamy okna 2-letniego (504 dni handlowe), aby określić co jest "Wysoko" a co "Nisko"
        window = 504
        
        # Funkcja pomocnicza do normalizacji MinMax w oknie rolowanym
        def normalize_rolling(series):
            return series.rolling(window).apply(
                lambda x: (x[-1] - x.min()) / (x.max() - x.min()) * 100 if (x.max() - x.min()) != 0 else 50, 
                raw=True
            )

        # Normalizujemy VIX (0=Min strach w ostatnich 2 latach, 100=Max strach)
        df['VIX_Rank'] = normalize_rolling(df['VIX'])
        
        # Normalizujemy Yield Curve (0=Najniższa/Inwersja, 100=Najwyższa/Stroma)
        df['Yield_Rank'] = normalize_rolling(df['Yield_Curve'])
        
        # Usuwamy puste wiersze na początku (brak danych do normalizacji)
        df.dropna(inplace=True)

        # 3. Obliczenia ANFIS
        anfis_engine.build_system()
        oscillator_values = []
        
        # Analizujemy ostatnie 5 lat (ok 1260 dni)
        df_analysis = df.tail(1260).copy()
        
        print("Rozpoczynanie analizy ANFIS...")
        for index, row in df_analysis.iterrows():
            score = anfis_engine.compute(
                rsi_val=row['RSI'],          # Technika
                vix_rank=row['VIX_Rank'],    # Sentyment
                yield_rank=row['Yield_Rank'] # Makro
            )
            oscillator_values.append(score)
            
        df_analysis['Sentiment_Oscillator'] = oscillator_values

        # 4. Formatowanie wyniku
        df_analysis.index.name = 'Date'
        df_analysis.reset_index(inplace=True)
        df_analysis['Date'] = df_analysis['Date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(df_analysis.to_dict(orient='list'))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    print("Uruchamianie serwera...")
    app.run(debug=True)