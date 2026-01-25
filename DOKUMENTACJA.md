# 📊 ANFIS Pro - Investor Assistant
## Dokumentacja Techniczna i Instrukcja Obsługi

**Wersja:** 2.0  
**Data:** Styczeń 2025  
**Autor:** System uczenia maszynowego oparty na logice rozmytej (ANFIS)

---

# Spis treści

1. [Wprowadzenie](#1-wprowadzenie)
2. [Architektura systemu](#2-architektura-systemu)
3. [Wymagania systemowe](#3-wymagania-systemowe)
4. [Instalacja](#4-instalacja)
5. [Struktura projektu](#5-struktura-projektu)
6. [Backend - Opis modułów](#6-backend---opis-modułów)
7. [Frontend - Opis komponentów](#7-frontend---opis-komponentów)
8. [API Reference](#8-api-reference)
9. [Instrukcja obsługi](#9-instrukcja-obsługi)
10. [Parametry i ich znaczenie](#10-parametry-i-ich-znaczenie)
11. [Interpretacja wyników](#11-interpretacja-wyników)
12. [Rozwiązywanie problemów](#12-rozwiązywanie-problemów)
13. [FAQ](#13-faq)

---

# 1. Wprowadzenie

## 1.1 Czym jest ANFIS Pro?

**ANFIS Pro - Investor Assistant** to zaawansowany system wspomagania decyzji inwestycyjnych oparty na **Adaptacyjnym Neuro-Rozmytym Systemie Wnioskowania (ANFIS)**. System łączy:

- **Logikę rozmytą** (Fuzzy Logic) - do modelowania niepewności rynkowej
- **Sieci neuronowe** (Neural Networks) - do uczenia się wzorców z danych historycznych
- **Wskaźniki techniczne** (RSI, MACD, VIX) - do analizy rynku
- **Dane makroekonomiczne** (Yield Curve, M2, Inflacja) - do szerszego kontekstu

## 1.2 Główne funkcje

| Funkcja | Opis |
|---------|------|
| **Analiza rynku** | Generowanie sygnałów kupna/sprzedaży na podstawie oscylatora ANFIS |
| **Uczenie maszynowe** | Trening modeli ANFIS do predykcji kierunku zmian cen |
| **Backtest** | Symulacja strategii na danych historycznych |
| **Mapa korelacji** | Analiza zależności między wskaźnikami a przyszłymi zwrotami |
| **Auto-strategia** | Automatyczne dobieranie najlepszych wskaźników przez AI |
| **Historia treningów** | Zapisywanie i porównywanie wyników eksperymentów |

## 1.3 Dla kogo jest ten system?

- **Inwestorzy indywidualni** - wspomaganie decyzji inwestycyjnych
- **Traderzy** - generowanie sygnałów technicznych
- **Badacze** - eksperymenty z modelami ANFIS
- **Studenci** - nauka o systemach rozmytych i ML w finansach

## 1.4 ⚠️ WAŻNE: FIS vs ANFIS - Wyjaśnienie terminologii

W projekcie używane są **dwa różne systemy**, które łatwo pomylić:

### 🔹 FIS (Fuzzy Inference System) - `fuzzy_expert_system.py`

**Co to jest:**
- Klasyczny **System Rozmyty** typu Mamdani
- **NIE MA uczenia maszynowego!**
- Reguły i funkcje przynależności zdefiniowane **ręcznie przez eksperta**
- Generuje oscylator sentymentu w czasie rzeczywistym

**Inne nazwy:** FLC (Fuzzy Logic Controller), Fuzzy Expert System

```
Człowiek definiuje:                  System oblicza:
┌─────────────────┐                  ┌─────────────────┐
│ MF: LOW=20      │                  │ Input: RSI=25   │
│      MED=50     │    ────────►     │ Output: 76      │
│      HIGH=80    │                  │ (sygnał kupna)  │
│                 │                  │                 │
│ Reguła: IF low  │                  │                 │
│ THEN buy        │                  │                 │
└─────────────────┘                  └─────────────────┘
   SZTYWNE!                            OBLICZONE
```

### 🔹 ANFIS (Adaptive Neuro-Fuzzy Inference System) - `anfis_ml_engine_v2.py`

**Co to jest:**
- **Prawdziwy ANFIS** - hybryda sieci neuronowej i logiki rozmytej
- **UCZY SIĘ** z danych przez gradient descent
- Parametry MF i reguł są **optymalizowane automatycznie**
- System Sugeno (TSK), nie Mamdani

```
Dane treningowe:                     Sieć neuronowa uczy się:
┌─────────────────┐                  ┌─────────────────┐
│ X: [RSI, MACD]  │                  │ MF centra: ?    │
│ Y: [returns]    │    ────────►     │ MF sigma: ?     │
│ ...1000 próbek  │   BACKPROP       │ Reguły: ?       │
│                 │                  │ Wagi: ?         │
└─────────────────┘                  └─────────────────┘
   DANE                                UCZONE!
```

### Porównanie:

| Cecha | FIS (fuzzy_expert_system.py) | ANFIS (anfis_ml_engine_v2.py) |
|-------|------------------------------|-------------------------------|
| **Uczenie** | ❌ Brak | ✅ Gradient descent |
| **MF** | Sztywne (20, 50, 80) | Uczone z danych |
| **Reguły** | Eksperckie IF-THEN | Automatyczne (n^k) |
| **Typ** | Mamdani | Sugeno (TSK) |
| **Szybkość** | Milisekundy | Minuty (trening) |
| **Użycie** | Oscylator real-time | Predykcja przyszłości |

---

# 2. Architektura systemu

## 2.1 Diagram architektury

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  App    │ │ Market  │ │Backtest │ │Correlat.│ │ANFIS ML │   │
│  │  .jsx   │ │ Chart   │ │ Panel   │ │ Panel   │ │ Panel   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (Flask/Python)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  app.py     │  │ Data Engine │  │ ANFIS ML Engine v2      │  │
│  │  (REST API) │  │ (yfinance)  │  │ (PyTorch - PRAWDZIWY    │  │
│  │             │  │             │  │  ANFIS z uczeniem!)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Backtest  │  │   Feature   │  │  FIS Mamdani (skfuzzy)  │  │
│  │   Engine    │  │   Factory   │  │  (NIE ANFIS! Bez uczenia│  │
│  │   Engine    │  │   Factory   │  │  (klasyczny oscylator)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ŹRÓDŁA DANYCH                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Yahoo      │  │   FRED API  │  │  Pliki lokalne (CSV)    │  │
│  │  Finance    │  │  (FED Data) │  │  training_history.csv   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 Przepływ danych

```
1. Użytkownik wybiera ticker (np. ^GSPC)
           │
           ▼
2. DataEngine pobiera dane:
   - Ceny z Yahoo Finance
   - Wskaźniki makro z FRED API
           │
           ▼
3. FeatureFactory generuje cechy:
   - Momentum (ROC)
   - Zmienność (Volatility)
   - Odchylenie od SMA
           │
           ▼
4. Normalizacja (ranking percentylowy 0-100)
           │
           ▼
5. ANFIS/ML przetwarza dane:
   - Funkcje przynależności (Gaussowskie)
   - Reguły rozmyte (IF-THEN)
   - Wyjście: Sentiment 0-100
           │
           ▼
6. Wyświetlenie na wykresie + sygnały
```

---

# 3. Wymagania systemowe

## 3.1 Minimalne wymagania

| Komponent | Wymaganie |
|-----------|-----------|
| **System operacyjny** | Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+) |
| **Python** | 3.8 - 3.11 (zalecane 3.10) |
| **Node.js** | 16.x lub nowszy |
| **RAM** | 4 GB minimum, 8 GB zalecane |
| **Dysk** | 500 MB wolnego miejsca |
| **Internet** | Wymagany (pobieranie danych) |

## 3.2 Opcjonalne (dla GPU)

| Komponent | Wymaganie |
|-----------|-----------|
| **CUDA** | 11.x lub 12.x |
| **GPU** | NVIDIA z min. 4GB VRAM |
| **PyTorch CUDA** | Odpowiednia wersja dla CUDA |

---

# 4. Instalacja

## 4.1 Instalacja backendu (Python)

### Krok 1: Utwórz środowisko wirtualne
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Krok 2: Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### Krok 3: Skonfiguruj klucz API FRED (opcjonalne)
Utwórz plik `config.py`:
```python
FRED_API_KEY = "twój_klucz_api"
```

Klucz możesz uzyskać za darmo na: https://fred.stlouisfed.org/docs/api/api_key.html

### Krok 4: Uruchom backend
```bash
python app.py
```
Backend uruchomi się na: `http://127.0.0.1:5000`

## 4.2 Instalacja frontendu (React)

### Krok 1: Utwórz projekt React (jeśli nie istnieje)
```bash
npm create vite@latest frontend -- --template react
cd frontend
```

### Krok 2: Zainstaluj zależności
```bash
npm install axios react-plotly.js plotly.js
```

### Krok 3: Skopiuj pliki JSX i CSS
Skopiuj wszystkie pliki `.jsx` i `.css` z paczki do folderu `src/`:
- App.jsx
- App.css
- MarketChart.jsx
- BacktestPanel.jsx
- CorrelationPanel.jsx
- InfoBadges.jsx
- CollapsibleSection.jsx
- AnfisMLPanel.jsx
- main.jsx
- index.css

### Krok 4: Uruchom frontend
```bash
npm run dev
```
Frontend uruchomi się na: `http://localhost:5173`

## 4.3 Weryfikacja instalacji

1. Otwórz przeglądarkę: `http://localhost:5173`
2. Wybierz ticker (np. ^GSPC)
3. Kliknij "Klasyczna" strategia
4. Kliknij "Analizuj Rynek"
5. Powinieneś zobaczyć wykres z danymi

---

# 5. Struktura projektu

```
anfis-pro/
│
├── backend/
│   ├── app.py                    # Główny serwer Flask (API)
│   ├── fuzzy_expert_system.py    # System rozmyty FIS Mamdani (NIE ANFIS!)
│   ├── weight_optimizer.py       # Optymalizator wag metodą DE
│   ├── anfis_ml_engine.py        # Silnik ML v1.0
│   ├── anfis_ml_engine_v2.py     # Silnik ML v2.0 (PRAWDZIWY ANFIS!)
│   ├── anfis_torch.py            # Implementacja ANFIS w PyTorch
│   ├── backtest_engine.py        # Silnik backtestingu
│   ├── data_engine.py            # Pobieranie danych (yfinance, FRED)
│   ├── feature_factory.py        # Generowanie cech
│   ├── config.py                 # Konfiguracja (API keys)
│   ├── requirements.txt          # Zależności Python
│   └── training_history.csv      # Historia treningów (auto-generowany)
│
├── frontend/src/
│   ├── App.jsx                   # Główny komponent aplikacji
│   ├── App.css                   # Style globalne
│   ├── MarketChart.jsx           # Wykres cen i oscylatora
│   ├── BacktestPanel.jsx         # Panel backtestingu
│   ├── CorrelationPanel.jsx      # Mapa korelacji (heatmapa)
│   ├── InfoBadges.jsx            # Kafelki wskaźników
│   ├── CollapsibleSection.jsx    # Zwijane sekcje
│   ├── AnfisMLPanel.jsx          # Panel uczenia maszynowego
│   ├── main.jsx                  # Entry point React
│   └── index.css                 # Style bazowe
│
└── docs/
    └── DOKUMENTACJA.md           # Ten plik
```

---

# 6. Backend - Opis modułów

## 6.1 app.py - Serwer REST API

Główny plik serwera Flask zawierający wszystkie endpointy API.

### Kluczowe endpointy:

| Endpoint | Metoda | Opis |
|----------|--------|------|
| `/api/tickers` | GET | Lista dostępnych instrumentów |
| `/api/analyze` | POST | Analiza rynku (oscylator ANFIS) |
| `/api/backtest` | POST | Symulacja strategii |
| `/api/correlations` | POST | Obliczenie korelacji |
| `/api/auto_strategy` | POST | Auto-dobór wskaźników |
| `/api/anfis_ml/train_stream` | POST | Trening ANFIS ML (streaming) |
| `/api/anfis_ml/save_results` | POST | Zapis wyników do CSV |
| `/api/anfis_ml/get_history` | GET | Pobranie historii treningów |

## 6.2 data_engine.py - Pobieranie danych

Moduł odpowiedzialny za pobieranie danych z zewnętrznych źródeł.

### Funkcje:

```python
class DataEngine:
    def get_market_data(ticker, period="max")
        """Pobiera dane giełdowe z Yahoo Finance"""
        # Zwraca: DataFrame z kolumnami [Price, RSI, MACD]
    
    def get_macro_data()
        """Pobiera dane makroekonomiczne z FRED API"""
        # Zwraca: DataFrame z kolumnami [Yield_Curve, VIX, M2_Liquidity, Inflation_CPI]
    
    def prepare_dataset(ticker)
        """Łączy dane rynkowe i makro"""
        # Zwraca: Połączony DataFrame
```

### Źródła danych:

| Źródło | Dane | Częstotliwość |
|--------|------|---------------|
| Yahoo Finance | Ceny, wolumen, RSI, MACD | Dzienne |
| FRED API (T10Y2Y) | Yield Curve (10Y-2Y) | Dzienne |
| FRED API (VIXCLS) | Indeks VIX (zmienność) | Dzienne |
| FRED API (M2SL) | Podaż pieniądza M2 | Miesięczne |
| FRED API (CPIAUCSL) | Inflacja CPI | Miesięczne |

## 6.3 feature_factory.py - Generowanie cech

Moduł tworzący wskaźniki techniczne z surowych danych.

### Generowane cechy:

Dla każdej kolumny bazowej (Price, RSI, VIX, MACD, M2) i okna czasowego (5, 10, 20, 50, 100 dni):

| Cecha | Wzór | Opis |
|-------|------|------|
| `{col}_ROC_{window}` | `(P[t] - P[t-n]) / P[t-n]` | Momentum (Rate of Change) |
| `{col}_Volat_{window}` | `std(P, window)` | Zmienność (odchylenie standardowe) |
| `{col}_DistSMA_{window}` | `(P - SMA) / SMA` | Odchylenie od średniej kroczącej |

**Przykłady wygenerowanych cech:**
- `Price_ROC_20` - 20-dniowe momentum ceny
- `RSI_Volat_50` - 50-dniowa zmienność RSI
- `VIX_DistSMA_100` - Odchylenie VIX od 100-dniowej średniej

## 6.4 fuzzy_expert_system.py - System Rozmyty FIS

> ⚠️ **WAŻNE:** Ten moduł to **NIE jest ANFIS!** To klasyczny **Fuzzy Inference System (FIS)** typu Mamdani.

### Czym jest FIS a czym ANFIS?

| Cecha | FIS (ten plik) | ANFIS (anfis_ml_engine_v2.py) |
|-------|----------------|------------------------------|
| **Uczenie** | ❌ Brak | ✅ Gradient descent |
| **Funkcje MF** | Sztywne (20, 50, 80) | Uczone z danych |
| **Reguły** | Zdefiniowane przez eksperta | Generowane automatycznie |
| **Typ systemu** | Mamdani | Sugeno (TSK) |
| **Architektura** | Klasyczne IF-THEN | Sieć neuronowa |

### Terminologia:

- **FIS** (Fuzzy Inference System) - system wnioskowania rozmytego
- **FLC** (Fuzzy Logic Controller) - kontroler logiki rozmytej
- **Mamdani** - typ FIS z rozmytym wyjściem
- **ANFIS** (Adaptive Neuro-Fuzzy Inference System) - hybryda sieci neuronowej i logiki rozmytej **Z UCZENIEM**

### Zasada działania FIS:

```
1. FUZZYFIKACJA (zamiana liczby na zbiory rozmyte)
   Input (0-100) → Funkcje przynależności (Low, Medium, High)
   
2. REGUŁY ROZMYTE (zdefiniowane przez eksperta - NIE UCZONE!)
   IF input IS high THEN sentiment IS buy
   IF input IS medium THEN sentiment IS neutral
   IF input IS low THEN sentiment IS sell
   
3. AGREGACJA I DEFUZZYFIKACJA
   Centroid method → Output (0-100)
```

### Funkcje przynależności (sztywne!):

```
         LOW        MEDIUM        HIGH
          │            │            │
    1.0 ──┼────────────┼────────────┼──
          │\          /│\          /│
          │ \        / │ \        / │
    0.5 ──│──\──────/──│──\──────/──│──
          │   \    /   │   \    /   │
          │    \  /    │    \  /    │
    0.0 ──│─────\/─────│─────\/─────│──
          0    20     50     80    100
          
    Centra: LOW=20, MEDIUM=50, HIGH=80 (SZTYWNE - nie uczą się!)
```

### Typy logiki:

| Typ | Interpretacja | Przykład |
|-----|---------------|----------|
| `pro_trend` | Wysoki input = BUY | MACD, M2 (rosnące = dobre) |
| `counter_trend` | Niski input = BUY | RSI, VIX (niskie = oversold = kupuj) |

### Do czego służy w projekcie:
- Generowanie **oscylatora sentymentu** (0-100) w czasie rzeczywistym
- Szybkie obliczenia (milisekundy)
- Wykres główny i backtest używają tego modułu

## 6.5 weight_optimizer.py - Optymalizator Wag

> ⚠️ **WAŻNE:** Ten moduł to **NIE jest uczenie ANFIS!** To optymalizacja meta-heurystyczna wag.

### Co robi:
- Szuka najlepszych **WAG** dla wskaźników (RSI, VIX, MACD, etc.)
- Używa algorytmu **Differential Evolution** (ewolucja różnicowa)
- **NIE modyfikuje** funkcji przynależności ani reguł!

### Jak działa:

```
1. Generuj populację zestawów wag
   [0.5, 1.2, 0.8, 1.0, 0.3]  ← osobnik 1
   [1.0, 0.5, 1.5, 0.2, 1.8]  ← osobnik 2
   
2. Dla każdego zestawu:
   → Zbuduj FIS z tymi wagami
   → Uruchom backtest
   → Oblicz zwrot (fitness)
   
3. Ewolucja: selekcja + mutacja + krzyżowanie
   
4. Powtórz przez N generacji
   
5. Zwróć najlepsze wagi
```

### Ryzyko:
**OVERFITTING!** Wagi dopasowane do przeszłości mogą nie działać w przyszłości.

## 6.6 anfis_ml_engine_v2.py - PRAWDZIWY ANFIS

Ten moduł implementuje **prawdziwy ANFIS** jako sieć neuronową w PyTorch.

### Kluczowa różnica:
**Wszystkie parametry są UCZONE przez gradient descent!**

### Architektura sieci:

```
       Input Layer (n features)
              │
              ▼
    ┌─────────────────────┐
    │  Fuzzification      │  ← Funkcje przynależności (Gauss/Bell/Tri)
    │  Layer              │     μ(x) = exp(-((x-c)/σ)²)
    │                     │     c, σ SĄ UCZONE!
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Rule Layer         │  ← Reguły T-norm (produkt)
    │  (AND operation)    │     w_i = μ_1 * μ_2 * ... * μ_n
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Normalization      │  ← Normalizacja wag
    │  Layer              │     w̄_i = w_i / Σw_j
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Consequent Layer   │  ← Funkcje liniowe (Sugeno)
    │  (TSK functions)    │     f_i = p_i*x₁ + q_i*x₂ + r_i
    │                     │     p, q, r SĄ UCZONE!
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Output Layer       │  ← Agregacja ważona
    │  (Defuzzification)  │     y = Σ(w̄_i * f_i)
    └─────────────────────┘
```

### Typy predykcji:

| Typ | Opis | Zastosowanie |
|-----|------|--------------|
| `returns` | % zmiana ceny | **Zalecane** - najlepsza interpretacja |
| `log_returns` | Logarytmiczna zmiana | Duże ruchy cenowe |
| `direction` | Kierunek (0/1) | Klasyfikacja binarna |
| `price` | Surowa cena | Niezalecane |

## 6.7 backtest_engine.py - Silnik symulacji

Moduł przeprowadzający backtest strategii na danych historycznych.

### Logika transakcji:

```python
# SYGNAŁ KUPNA
if score >= buy_threshold and not in_market:
    BUY()

# SYGNAŁ SPRZEDAŻY
if in_market:
    if score <= sell_threshold:    # Sygnał FIS
        SELL()
    elif price <= entry * (1 - stop_loss):  # Stop Loss
        SELL()
    elif price >= entry * (1 + take_profit):  # Take Profit
        SELL()
    elif trailing_stop and price <= max_price * (1 - trail_pct):
        SELL()  # Trailing Stop
```

### Metryki wynikowe:

| Metryka | Opis | Dobra wartość |
|---------|------|---------------|
| Total Return | Całkowity zwrot (%) | > Buy & Hold |
| Sharpe Ratio | Zwrot/Ryzyko | > 1.0 |
| Max Drawdown | Maksymalny spadek (%) | < 20% |
| Win Rate | % zyskownych transakcji | > 50% |

---

# 7. Frontend - Opis komponentów

## 7.1 App.jsx - Główny komponent

Zarządza stanem aplikacji i układem paneli.

### Stan główny:
```javascript
const [ticker, setTicker] = useState('^GSPC');      // Wybrany instrument
const [chartData, setChartData] = useState(null);   // Dane wykresu
const [indicatorsConfig, setIndicatorsConfig] = useState({});  // Konfiguracja wskaźników
```

## 7.2 MarketChart.jsx - Wykres analityczny

Wyświetla dwupanelowy wykres Plotly:
- **Górny panel:** Cena instrumentu
- **Dolny panel:** Oscylator ANFIS (0-100) + wskaźniki pomocnicze

### Strefy na oscylatorze:
- **> 80:** Strefa wykupienia (SELL)
- **50:** Neutralna
- **< 20:** Strefa wyprzedania (BUY)

## 7.3 InfoBadges.jsx - Panel wskaźników

Wyświetla kafelki z aktywnymi wskaźnikami i suwakami wag.

### Funkcje:
- **Klasyczna strategia:** RSI, VIX, MACD, Yield Curve, M2
- **Generuj (AI):** Automatyczny dobór na podstawie korelacji

## 7.4 CorrelationPanel.jsx - Mapa korelacji

Heatmapa pokazująca korelacje między wskaźnikami a przyszłymi zwrotami.

### Metody korelacji:
| Metoda | Opis |
|--------|------|
| Pearson | Liniowa zależność |
| Spearman | Rangowa (odporna na outliers) |
| Kendall | Zgodność par (najbardziej konserwatywna) |

### Interpretacja kolorów:
- 🟢 **Zielony (+):** Wysoka wartość = wzrost ceny
- 🔴 **Czerwony (-):** Wysoka wartość = spadek ceny
- ⚫ **Ciemny (0):** Brak korelacji

## 7.5 BacktestPanel.jsx - Panel symulacji

Umożliwia testowanie strategii na danych historycznych.

### Parametry:
| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| Kup ≥ | 65 | Próg sygnału kupna |
| Sprzedaj ≤ | 35 | Próg sygnału sprzedaży |
| SL (%) | 5 | Stop Loss |
| TP (%) | 0 | Take Profit (0 = wyłączony) |
| Trailing | Off | Trailing Stop |

## 7.6 AnfisMLPanel.jsx - Panel uczenia maszynowego

Najbardziej zaawansowany panel - trening modeli ANFIS.

### Sekcje:
1. **Parametry treningu** - konfiguracja modelu
2. **Wyniki** - metryki i wykresy
3. **Zakładki** - Predykcje, Scatter, Loss, MF, Ważność, Reguły, Debug
4. **Historia treningów** - zapisane eksperymenty

---

# 8. API Reference

## 8.1 GET /api/tickers

Zwraca listę dostępnych instrumentów.

**Response:**
```json
[
  {"symbol": "^GSPC", "name": "S&P 500 (USA)"},
  {"symbol": "^NDX", "name": "Nasdaq 100 (USA)"},
  {"symbol": "BTC-USD", "name": "Bitcoin / USD"},
  ...
]
```

## 8.2 POST /api/analyze

Wykonuje analizę rynku i zwraca dane do wykresu.

**Request:**
```json
{
  "ticker": "^GSPC",
  "config": {
    "RSI": {"enabled": true, "weight": 1.0, "direction": -1},
    "VIX": {"enabled": true, "weight": 1.0, "direction": 1}
  }
}
```

**Response:**
```json
{
  "Date": ["2024-01-01", "2024-01-02", ...],
  "Price": [4800.5, 4815.2, ...],
  "Sentiment_Oscillator": [55.2, 58.1, ...],
  "RSI": [45.2, 48.1, ...],
  "VIX": [12.5, 13.2, ...]
}
```

## 8.3 POST /api/backtest

Wykonuje backtest strategii.

**Request:**
```json
{
  "ticker": "^GSPC",
  "config": {...},
  "buyThreshold": 70,
  "sellThreshold": 30,
  "stopLoss": 5,
  "takeProfit": 15,
  "trailingStop": false
}
```

**Response:**
```json
{
  "initial_capital": 10000,
  "final_value": 12500.50,
  "total_return": 25.05,
  "buy_and_hold_return": 18.20,
  "trades_count": 15,
  "max_drawdown": -8.5,
  "win_rate": 60.0,
  "sharpe_ratio": 1.45,
  "trades": [...],
  "equity_curve": [...],
  "bnh_curve": [...]
}
```

## 8.4 POST /api/anfis_ml/train_stream

Trenuje model ANFIS ML (streaming response).

**Request:**
```json
{
  "ticker": "^GSPC",
  "config": {...},
  "epochs": 100,
  "num_mfs": 3,
  "batch_size": 64,
  "learning_rate": 0.01,
  "mf_type": "gauss",
  "hybrid": true,
  "optimizer": "adam",
  "lookahead": 1,
  "prediction_type": "returns",
  "scaler_type": "robust",
  "early_stopping_patience": 20,
  "training_days": 0
}
```

**Response (streaming):**
```json
{"status": "training", "epoch": 1, "progress": 1, "val_rmse": 0.0234}
{"status": "training", "epoch": 2, "progress": 2, "val_rmse": 0.0198}
...
{"status": "done", "metrics": {...}, "predictions": {...}}
```

## 8.5 POST /api/anfis_ml/save_results

Zapisuje wyniki treningu do CSV.

**Request:**
```json
{
  "ticker": "^GSPC",
  "prediction_type": "returns",
  "epochs": 100,
  "direction_accuracy": 58.5,
  "rmse": 0.0123,
  "notes": "Test z RSI i VIX"
}
```

## 8.6 GET /api/anfis_ml/get_history

Pobiera historię treningów.

**Response:**
```json
{
  "history": [
    {
      "id": "a1b2c3d4",
      "timestamp": "2025-01-15 14:30:00",
      "ticker": "^GSPC",
      "direction_accuracy": 58.5,
      ...
    }
  ],
  "total_records": 10,
  "stats": {
    "avg_direction_accuracy": 54.2,
    "max_direction_accuracy": 62.1
  }
}
```

## 8.7 DELETE /api/anfis_ml/delete_history

Usuwa wybrane lub wszystkie rekordy.

**Request:**
```json
{
  "ids": ["a1b2c3d4", "e5f6g7h8"]  // lub [] dla usunięcia wszystkiego
}
```

## 8.8 POST /api/anfis_ml/export_selected

Eksportuje wybrane rekordy jako CSV.

**Request:**
```json
{
  "ids": ["a1b2c3d4"]  // lub [] dla eksportu wszystkiego
}
```

---

# 9. Instrukcja obsługi

## 9.1 Szybki start (Quick Start)

### Krok 1: Uruchom aplikację
1. Uruchom backend: `python app.py`
2. Uruchom frontend: `npm run dev`
3. Otwórz: `http://localhost:5173`

### Krok 2: Wybierz instrument
- Wpisz symbol (np. `^GSPC`, `BTC-USD`) lub wybierz z listy

### Krok 3: Wybierz strategię
- Kliknij **"Klasyczna"** dla domyślnych wskaźników
- Lub kliknij **"Generuj (AI)"** dla automatycznego doboru

### Krok 4: Analizuj
- Kliknij **"Analizuj Rynek"**
- Obserwuj wykres ceny i oscylatora ANFIS

### Krok 5: Przetestuj strategię
- Rozwiń sekcję **"Symulator Strategii (Backtest)"**
- Ustaw progi i parametry
- Kliknij **"Start"**

## 9.2 Analiza rynku

### Interpretacja oscylatora ANFIS:

```
100 ┬─────────────── SILNE WYKUPIENIE (SELL) ───────────────┐
    │                    Strefa czerwona                     │
 80 ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
    │                                                        │
 50 ├─────────────────── NEUTRALNY ──────────────────────────┤
    │                                                        │
 20 ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
    │                    Strefa zielona                      │
  0 └─────────────── SILNE WYPRZEDANIE (BUY) ───────────────┘
```

### Sygnały handlowe:
- **BUY:** Oscylator < 30 i rośnie
- **SELL:** Oscylator > 70 i spada
- **HOLD:** Oscylator w strefie neutralnej (30-70)

## 9.3 Mapa korelacji

### Jak korzystać:
1. Rozwiń sekcję **"Mapa Korelacji"**
2. Poczekaj na wygenerowanie (może potrwać ~10s)
3. Analizuj heatmapę

### Interpretacja:
- Szukaj cech z **silną dodatnią korelacją** (zielone) - wzrost cechy = wzrost ceny
- Szukaj cech z **silną ujemną korelacją** (czerwone) - wzrost cechy = spadek ceny
- Unikaj cech z **zerową korelacją** (ciemne) - brak predykcyjności

### Wskazówka:
Kliknij **"Generuj (AI)"** aby automatycznie wybrać najlepsze cechy z mapy korelacji.

## 9.4 Backtest strategii

### Krok 1: Ustaw parametry
| Parametr | Opis | Zalecane |
|----------|------|----------|
| Kup ≥ | Próg kupna (oscylator) | 60-70 |
| Sprzedaj ≤ | Próg sprzedaży | 30-40 |
| SL | Stop Loss (% straty) | 3-5% |
| TP | Take Profit (% zysku) | 10-20% lub 0 |
| Trailing | Trailing Stop | Dla trendów |

### Krok 2: Uruchom symulację
- Kliknij **"Start"**
- Obserwuj wyniki

### Krok 3: Analiza wyników
- **Strategia vs Rynek:** Czy strategia pokonuje Buy & Hold?
- **Sharpe Ratio:** > 1.0 oznacza dobry stosunek zysk/ryzyko
- **Max Drawdown:** < 20% oznacza akceptowalne ryzyko
- **Win Rate:** > 50% z odpowiednim R:R

### Krok 4: Optymalizacja
- Zmieniaj parametry i testuj ponownie
- Zapisuj najlepsze konfiguracje

## 9.5 Uczenie maszynowe ANFIS

### Krok 1: Skonfiguruj cechy
- Wybierz wskaźniki (RSI, VIX, MACD, itp.)
- Lub użyj **"Generuj (AI)"**

### Krok 2: Ustaw parametry treningu

| Parametr | Zalecane | Opis |
|----------|----------|------|
| Typ predykcji | `% Zmiana Ceny` | Najlepsza interpretacja |
| Epoki | 100-200 | Więcej = dłuższy trening |
| Liczba MF | 3 | 2-5, więcej = bardziej złożony model |
| Typ MF | Gaussowska | Najlepsza dla danych ciągłych |
| Optymalizator | Adam | Najszybszy i stabilny |
| Learning Rate | 0.01 | 0.001-0.05 |
| Batch Size | 64 | 32-128 |
| Horyzont | 1-5 dni | Ile dni w przód predykcja |
| Hybrid (LSE) | ✓ Tak | Szybsza konwergencja |

### Krok 3: Uruchom trening
- Kliknij **"Rozpocznij Uczenie ANFIS"**
- Obserwuj postęp i metryki na żywo

### Krok 4: Analiza wyników
- **Direction Accuracy > 55%** = model użyteczny
- **Direction Accuracy > 60%** = model bardzo dobry
- Sprawdź zakładki: Predykcje, Scatter, Loss, MF

### Krok 5: Zapisz wyniki
- Dodaj notatki (opcjonalnie)
- Kliknij **"Zapisz do CSV"**

## 9.6 Historia treningów

### Przeglądanie:
- Każdy rekord to rozwijalny panel
- Kliknij na rekord aby zobaczyć szczegóły
- Użyj **"Rozwiń wszystkie"** dla porównania

### Selekcja:
- Zaznacz checkboxy przy rekordach
- Lub użyj **"Zaznacz wszystko"**

### Akcje:
- **Eksport:** Pobierz zaznaczone jako CSV
- **Usuń:** Usuń zaznaczone rekordy
- **Odśwież:** Załaduj ponownie historię

---

# 10. Parametry i ich znaczenie

## 10.1 Wskaźniki techniczne

### RSI (Relative Strength Index)
| Wartość | Interpretacja | Sygnał |
|---------|---------------|--------|
| < 30 | Wyprzedanie | BUY |
| 30-70 | Neutralny | HOLD |
| > 70 | Wykupienie | SELL |

**Kierunek:** Counter-trend (-1) - niski RSI = sygnał kupna

### MACD (Histogram)
| Wartość | Interpretacja | Sygnał |
|---------|---------------|--------|
| > 0 (rosnący) | Momentum wzrostowe | BUY |
| < 0 (spadający) | Momentum spadkowe | SELL |

**Kierunek:** Pro-trend (+1) - wysoki MACD = sygnał kupna

### VIX (Indeks zmienności)
| Wartość | Interpretacja | Sygnał |
|---------|---------------|--------|
| < 15 | Niski strach | HOLD |
| 15-25 | Normalny | HOLD |
| > 25 | Wysoki strach | BUY (contrarian) |

**Kierunek:** Counter-trend (-1) - wysoki VIX = okazja kupna

### Yield Curve (T10Y2Y)
| Wartość | Interpretacja | Sygnał |
|---------|---------------|--------|
| > 0 | Normalna krzywa | Pozytywny |
| < 0 | Inwersja | Negatywny (recesja) |

**Kierunek:** Pro-trend (+1) - pozytywna krzywa = dobry sygnał

### M2 Liquidity (YoY)
| Wartość | Interpretacja | Sygnał |
|---------|---------------|--------|
| > 5% | Ekspansja monetarna | BUY |
| 0-5% | Normalny | HOLD |
| < 0% | Kontrakcja | SELL |

**Kierunek:** Pro-trend (+1) - wzrost płynności = dobry sygnał

## 10.2 Parametry treningu ANFIS

### Typ predykcji

| Typ | Wzór | Zalecenia |
|-----|------|-----------|
| `% Zmiana Ceny` | `(P[t+n]/P[t] - 1) * 100` | **Zalecane** - łatwa interpretacja |
| `Log Returns` | `ln(P[t+n]/P[t]) * 100` | Duże ruchy, symetryczne |
| `Kierunek` | `1 jeśli P[t+n] > P[t]` | Klasyfikacja binarna |
| `Surowa Cena` | `P[t+n]` | Niezalecane - niestacjonarne |

### Liczba funkcji przynależności (MF)

| Wartość | Złożoność | Ryzyko | Zastosowanie |
|---------|-----------|--------|--------------|
| 2 | Niska | Underfitting | Szybkie testy |
| 3 | Średnia | Optymalnie | **Zalecane** |
| 4-5 | Wysoka | Overfitting | Dużo danych |

### Typ funkcji przynależności

| Typ | Kształt | Zalecenia |
|-----|---------|-----------|
| Gaussowska | Dzwonowa | **Zalecane** - gładka, naturalny rozkład |
| Dzwonowa (Bell) | Szeroka | Więcej parametrów |
| Trójkątna | Ostre krawędzie | Szybka, mniej dokładna |

### Optymalizator

| Typ | Charakterystyka | Zalecenia |
|-----|-----------------|-----------|
| Adam | Adaptacyjny LR | **Zalecane** - szybki, stabilny |
| AdamW | Adam + weight decay | Lepsza regularyzacja |
| SGD | Klasyczny | Wolniejszy, stabilniejszy |

### Learning Rate

| Wartość | Efekt |
|---------|-------|
| 0.001 | Wolne uczenie, stabilne |
| 0.01 | **Zalecane** - balans |
| 0.05 | Szybkie, ryzyko niestabilności |

### Batch Size

| Wartość | Efekt |
|---------|-------|
| 32 | Więcej szumu, lepsza generalizacja |
| 64 | **Zalecane** - balans |
| 128 | Stabilniejsze gradienty, może overfittować |

### Early Stopping Patience

| Wartość | Efekt |
|---------|-------|
| 10 | Szybkie zatrzymanie, ryzyko underfitting |
| 20 | **Zalecane** |
| 50 | Dłuższy trening, ryzyko overfitting |

### Dni treningowe

| Wartość | Dane | Zalecenia |
|---------|------|-----------|
| Wszystkie | Cała historia | Więcej danych |
| 5 lat (1260) | ~5 lat | **Zalecane** - nowsze wzorce |
| 2 lata (504) | ~2 lata | Szybki test |

---

# 11. Interpretacja wyników

## 11.1 Metryki modelu ANFIS ML

### Direction Accuracy (Celność Kierunku)

**Co to jest:** Procent poprawnie przewidzianych kierunków zmian ceny.

| Wartość | Interpretacja | Akcja |
|---------|---------------|-------|
| < 50% | Gorsze niż losowe | Odrzuć model |
| 50-52% | Na poziomie losowym | Nieużyteczny |
| 52-55% | Lekka przewaga | Można testować |
| **55-60%** | **Dobry model** | **Wart użycia** |
| > 60% | Bardzo dobry | Sprawdź overfitting |

### RMSE (Root Mean Square Error)

**Co to jest:** Średni błąd predykcji (w jednostkach target).

| Dla `% Zmiana Ceny` | Interpretacja |
|---------------------|---------------|
| < 1.0 | Bardzo dobry |
| 1.0 - 2.0 | Dobry |
| > 2.0 | Słaby |

### R² (Współczynnik determinacji)

**Co to jest:** Ile % wariancji wyjaśnia model.

| Wartość | Interpretacja |
|---------|---------------|
| < 0% | Model gorszy niż średnia |
| 0-10% | Słaby model |
| 10-30% | Dobry dla finansów |
| > 30% | Bardzo dobry (sprawdź overfitting) |

### Korelacja

**Co to jest:** Zależność liniowa między predykcją a rzeczywistością.

| Wartość | Interpretacja |
|---------|---------------|
| < 0.1 | Brak zależności |
| 0.1-0.3 | Słaba |
| 0.3-0.5 | Umiarkowana |
| > 0.5 | Silna |

## 11.2 Metryki backtestingu

### Total Return vs Buy & Hold

| Scenariusz | Interpretacja |
|------------|---------------|
| Strategia > B&H | Strategia działa |
| Strategia ≈ B&H | Neutralna |
| Strategia < B&H | Strategia nie działa |

### Sharpe Ratio

**Wzór:** `(Return - RiskFreeRate) / StandardDeviation`

| Wartość | Interpretacja |
|---------|---------------|
| < 0 | Strata |
| 0-1 | Słaby |
| 1-2 | Dobry |
| > 2 | Bardzo dobry |

### Max Drawdown

**Co to jest:** Największy spadek od szczytu do dołka.

| Wartość | Interpretacja |
|---------|---------------|
| < 10% | Niskie ryzyko |
| 10-20% | Akceptowalne |
| 20-30% | Wysokie ryzyko |
| > 30% | Bardzo wysokie ryzyko |

### Win Rate

**Co to jest:** % zyskownych transakcji.

| Wartość | Interpretacja |
|---------|---------------|
| < 40% | Słabe (chyba że duży R:R) |
| 40-50% | Akceptowalne z dobrym R:R |
| 50-60% | Dobre |
| > 60% | Bardzo dobre |

## 11.3 Interpretacja wykresów

### Wykres predykcji (zakładka "Predykcje")

```
         Rzeczywiste (niebieska linia)
              │
              ▼
    ─────╱╲──────╱╲──────╱╲─────
        ╱  ╲    ╱  ╲    ╱  ╲
       ╱    ╲  ╱    ╲  ╱    ╲
    ──╱──────╲╱──────╲╱──────╲──
              ▲
              │
         Predykcja (zielona przerywana)
```

**Dobry model:** Linie podążają za sobą, szczególnie w kierunkach

### Scatter plot (zakładka "Scatter")

```
    Predykcja
        ▲
        │    • • •
        │  • • • • •
        │• • • • • • •
    ────┼───────────────► Rzeczywiste
        │
```

**Dobry model:** Punkty blisko linii diagonalnej

**Kolory punktów:**
- 🟢 Zielony = poprawny kierunek
- 🔴 Czerwony = błędny kierunek

### Wykres Loss (zakładka "Loss")

```
    Loss
      │╲
      │ ╲
      │  ╲___Train (niebieski)
      │     ╲___Val (czerwony)
      │         ────────────
      └──────────────────────► Epoki
```

**Dobry trening:**
- Obie linie maleją
- Val loss nie rośnie (brak overfitting)
- Stabilizacja pod koniec

---

# 12. Rozwiązywanie problemów

## 12.1 Backend nie uruchamia się

### Problem: `ModuleNotFoundError`
```
ModuleNotFoundError: No module named 'torch'
```

**Rozwiązanie:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Problem: `FRED API error`
```
⚠️ Błąd pobierania T10Y2Y (Direct): ...
```

**Rozwiązanie:**
1. Utwórz plik `config.py` z kluczem API
2. Lub system użyje domyślnych wartości (0)

### Problem: `Port 5000 in use`
```
OSError: [Errno 98] Address already in use
```

**Rozwiązanie:**
```bash
# Linux/Mac
kill $(lsof -t -i:5000)

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

## 12.2 Frontend nie łączy się z backendem

### Problem: `Network Error`
```
Error: Network Error
```

**Rozwiązanie:**
1. Sprawdź czy backend działa: `http://127.0.0.1:5000/api/tickers`
2. Sprawdź CORS w `app.py` (powinno być `CORS(app)`)
3. Wyłącz VPN/firewall

### Problem: `CORS error`
```
Access to fetch blocked by CORS policy
```

**Rozwiązanie:**
Upewnij się, że w `app.py`:
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
```

## 12.3 Trening nie działa

### Problem: `No active features found`
```
Error: No active features found in data
```

**Rozwiązanie:**
- Wybierz przynajmniej jeden wskaźnik (RSI, VIX, itp.)
- Kliknij "Klasyczna" lub "Generuj (AI)"

### Problem: `CUDA out of memory`
```
RuntimeError: CUDA out of memory
```

**Rozwiązanie:**
- Zmniejsz `Batch Size` (np. 32)
- Zmniejsz `Liczba MF` (np. 2)
- Lub użyj CPU (automatycznie gdy brak GPU)

### Problem: Trening trwa bardzo długo
**Rozwiązanie:**
- Zmniejsz liczbę epok (np. 50)
- Zmniejsz dni treningowe (np. 2 lata)
- Użyj mniej cech

## 12.4 Historia treningów pusta

### Problem: Tabela historii pusta mimo zapisanych treningów

**Rozwiązanie:**
1. Sprawdź konsolę backendu - powinno być:
   ```
   📂 Szukam pliku: /path/to/training_history.csv
   📊 Wczytano X rekordów z CSV
   ```
2. Sprawdź czy plik `training_history.csv` istnieje
3. Kliknij "Odśwież" w sekcji historii

## 12.5 Wyniki są słabe

### Problem: Direction Accuracy < 50%

**Rozwiązanie:**
1. Zmień typ predykcji na `% Zmiana Ceny`
2. Użyj innych cech (sprawdź mapę korelacji)
3. Zmień horyzont (np. 5 dni zamiast 1)
4. Użyj więcej danych treningowych

### Problem: Model overfittuje (Train loss << Val loss)

**Rozwiązanie:**
1. Zmniejsz liczbę MF (np. 2)
2. Zmniejsz liczbę epok
3. Zwiększ Early Stopping Patience
4. Użyj mniej cech

---

# 13. FAQ

## Pytania ogólne

**Q: Czy system nadaje się do realnego tradingu?**
A: System jest narzędziem wspomagającym, nie gwarancją zysków. Zawsze testuj na danych historycznych i zacznij od małych pozycji.

**Q: Czy potrzebuję GPU?**
A: Nie, CPU wystarczy. GPU przyspiesza trening ~5-10x, ale nie jest konieczne.

**Q: Jak często aktualizować model?**
A: Zalecane co 1-3 miesiące lub gdy rynek znacząco się zmieni.

## Pytania techniczne

**Q: Dlaczego Direction Accuracy oscyluje wokół 50%?**
A: Rynki finansowe są trudne do przewidzenia. 55%+ to już dobry wynik. Sprawdź:
- Czy używasz odpowiednich cech (mapa korelacji)
- Czy typ predykcji to `% Zmiana Ceny`
- Czy dane nie są zbyt zaszumione

**Q: Co oznacza "Hybrid (LSE)"?**
A: Least Squares Estimation - metoda optymalizacji warstwy wyjściowej ANFIS. Przyspiesza uczenie i poprawia stabilność.

**Q: Ile cech powinienem używać?**
A: 3-7 cech. Zbyt mało = underfitting, zbyt dużo = overfitting i długi trening.

**Q: Jak interpretować funkcje przynależności (zakładka MF)?**
A: Każdy wykres pokazuje jak model "widzi" daną cechę:
- Krzywe pokazują stopień przynależności do kategorii (Low, Medium, High)
- Nakładające się krzywe = płynne przejścia
- Wąskie krzywe = ostrzejsze decyzje

## Pytania o dane

**Q: Skąd pochodzą dane?**
A: 
- Ceny: Yahoo Finance (darmowe, opóźnione ~15 min)
- Makro: FRED API (darmowe z kluczem)

**Q: Czy mogę używać własnych danych?**
A: Tak, można zmodyfikować `data_engine.py` aby wczytywać dane z CSV.

**Q: Dlaczego brakuje danych makro?**
A: Prawdopodobnie brak klucza API FRED. Utwórz `config.py` z kluczem.

---

# Załączniki

## A. Wzory matematyczne

### Funkcja przynależności Gaussowska
```
μ(x) = exp(-((x - c)² / 2σ²))
```
gdzie: c = centrum, σ = szerokość

### Reguła T-norm (AND)
```
w_i = μ₁(x₁) × μ₂(x₂) × ... × μₙ(xₙ)
```

### Normalizacja wag
```
w̄_i = w_i / Σⱼw_j
```

### Wyjście ANFIS (Sugeno pierwszego rzędu)
```
y = Σᵢ(w̄_i × fᵢ)

gdzie fᵢ = pᵢ×x₁ + qᵢ×x₂ + ... + rᵢ
```

### Sharpe Ratio
```
SR = (E[R] - Rf) / σ[R]
```
gdzie: E[R] = średni zwrot, Rf = stopa wolna od ryzyka, σ = odchylenie standardowe

### Maximum Drawdown
```
MDD = max(Peak - Trough) / Peak × 100%
```

## B. Skróty i definicje

| Skrót | Pełna nazwa | Opis |
|-------|-------------|------|
| ANFIS | Adaptive Neuro-Fuzzy Inference System | System hybrydowy łączący sieci neuronowe i logikę rozmytą |
| MF | Membership Function | Funkcja przynależności |
| RSI | Relative Strength Index | Wskaźnik siły względnej |
| MACD | Moving Average Convergence Divergence | Wskaźnik momentum |
| VIX | Volatility Index | Indeks zmienności (strach) |
| ROC | Rate of Change | Stopa zmiany (momentum) |
| SMA | Simple Moving Average | Prosta średnia krocząca |
| LSE | Least Squares Estimation | Estymacja najmniejszych kwadratów |
| B&H | Buy and Hold | Strategia "kup i trzymaj" |
| SL | Stop Loss | Zlecenie ochronne ograniczające stratę |
| TP | Take Profit | Zlecenie realizujące zysk |

## C. Zalecane konfiguracje

### Dla początkujących (S&P 500)
```
Ticker: ^GSPC
Strategia: Klasyczna
Typ predykcji: % Zmiana Ceny
Epoki: 100
MF: 3 (Gauss)
LR: 0.01
Horyzont: 1 dzień
```

### Dla kryptowalut (Bitcoin)
```
Ticker: BTC-USD
Strategia: AI (wysoka zmienność)
Typ predykcji: Log Returns
Epoki: 150
MF: 4 (Gauss)
LR: 0.005
Horyzont: 1-3 dni
```

### Dla długoterminowych inwestorów
```
Ticker: ^GSPC
Cechy: VIX, Yield Curve, M2
Typ predykcji: % Zmiana Ceny
Epoki: 200
MF: 3 (Gauss)
Horyzont: 5-10 dni
Backtest: SL=10%, TP=0 (trend following)
```

---

**Koniec dokumentacji**

*ANFIS Pro - Investor Assistant v2.0*  
*© 2025 - Dokumentacja wygenerowana automatycznie*
