from flask import Flask, render_template, jsonify, request
from data_engine import DataEngine
from anfis_engine import AnfisEngine
import pandas as pd
import json

app = Flask(__name__)

# --- KONFIGURACJA ---
# PAMIĘTAJ: Wklej tutaj swój klucz API z fred.stlouisfed.org
FRED_API_KEY = 'b7d804e08b899c4a8c9fdfff48dfdad8pytho' 

data_engine = DataEngine(FRED_API_KEY)
anfis_engine = AnfisEngine()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    req_data = request.json
    ticker = req_data.get('ticker', '^GSPC')
    
    # 1. Pobieranie danych
    try:
        # Pobieramy dane rynkowe i makro
        # Domyślny zestaw wskaźników makro do korelacji
        macro_config = {
            'M2SL': 'yoy',        # Podaż pieniądza (zmiana roczna)
            'FEDFUNDS': 'raw',    # Stopy procentowe
            'CPIAUCSL': 'yoy',    # Inflacja
            'UNRATE': 'raw',      # Bezrobocie
            'T10Y2Y': 'raw'       # Krzywa dochodowości
        }
        
        df = data_engine.prepare_dataset(ticker, macro_config)
        
        if df.empty:
            return jsonify({"error": "Brak danych dla podanych parametrów."}), 400

        # 2. Analiza ANFIS (Generowanie Oscylatora Nastrojów)
        # Dla wydajności obliczamy to tylko dla ostatnich 200 dni w tym demo
        df_analysis = df.tail(200).copy()
        sentiments = []
        
        # Budujemy system rozmyty
        anfis_engine.build_system()
        
        for index, row in df_analysis.iterrows():
            # Przekazujemy Inflację i Bezrobocie do systemu ANFIS
            # W prawdziwej aplikacji te wejścia byłyby dynamicznie wybierane przez użytkownika
            val = anfis_engine.compute_sentiment(
                inflation_val=row.get('CPIAUCSL', 2.0), 
                unemployment_val=row.get('UNRATE', 4.0)
            )
            sentiments.append(val)
            
        df_analysis['Sentiment_Oscillator'] = sentiments

        # 3. Przygotowanie odpowiedzi JSON
        df_analysis.reset_index(inplace=True)
        df_analysis['Date'] = df_analysis['Date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(df_analysis.to_dict(orient='list'))

    except Exception as e:
        print(f"Błąd: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Uruchamianie serwera...")
    print("Pamiętaj o dodaniu klucza FRED API w pliku app.py!")
    app.run(debug=True)