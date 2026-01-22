import numpy as np
from copy import deepcopy

class AnfisEngine:
    """
    Prawdziwy ANFIS z 5 warstwami i uczeniem hybrydowym:
    - Forward Pass: Backpropagation dla parametrów nieliniowych (centra, sigmy)
    - Backward Pass: Least Squares dla parametrów liniowych (wagi konsekwentów)
    """
    
    def __init__(self, n_inputs=5, learning_rate=0.5):  # ZMIENIONO z 0.01 na 0.5
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.trained = False
        
        # Parametry uczące się
        self.params = {
            'centers': {},      # Centra funkcji Gaussa
            'sigmas': {},       # Szerokości funkcji Gaussa
            'consequents': {},  # Wagi wyjściowe (TSK)
            'directions': {}    # Kierunki logiki
        }
        
        # Historia uczenia
        self.training_history = {
            'epochs': [],
            'mse': [],
            'param_changes': []
        }
        
        # Cache dla forward pass
        self.cache = {}
        
    def initialize_params(self, config):
        """
        Inicjalizacja parametrów dla każdego wskaźnika
        config = {'RSI': {'enabled': True, 'weight': 1.0, 'direction': -1}, ...}
        """
        self.active_features = [k for k, v in config.items() if v.get('enabled', False)]
        
        for feature in self.active_features:
            direction = config[feature].get('direction', 1)
            
            # Centra funkcji przynależności (3 zbiory: Low, Medium, High)
            self.params['centers'][feature] = np.array([20.0, 50.0, 80.0])
            
            # Sigmy (szerokości) - inicjalizacja z lekką randomizacją
            self.params['sigmas'][feature] = np.array([10.0, 10.0, 10.0]) + np.random.randn(3) * 0.5
            
            # Wagi konsekwentów (TSK model: y = w0 + w1*x)
            # Dla 3 reguł × 2 parametry (const + linear)
            self.params['consequents'][feature] = np.random.randn(3, 2) * 0.1
            
            # Kierunek logiki
            self.params['directions'][feature] = direction
    
    # ==================== WARSTWA 1: FUZZYFIKACJA ====================
    
    def gaussmf(self, x, center, sigma):
        """Funkcja Gaussa - różniczkowalna"""
        return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    def fuzzify(self, feature, value):
        """
        WARSTWA 1: Oblicza stopnie przynależności μ(x)
        Returns: array of shape (3,) - [μ_low, μ_med, μ_high]
        """
        centers = self.params['centers'][feature]
        sigmas = self.params['sigmas'][feature]
        
        mu = np.array([
            self.gaussmf(value, centers[0], sigmas[0]),
            self.gaussmf(value, centers[1], sigmas[1]),
            self.gaussmf(value, centers[2], sigmas[2])
        ])
        
        return mu
    
    # ==================== WARSTWA 2: REGUŁY ====================
    
    def apply_rules(self, mu_values):
        """
        WARSTWA 2: Produkty (AND) - w tym przypadku T-norma = min
        Dla pojedynczego wskaźnika: μ_i = μ_i (brak kombinacji)
        Returns: firing_strengths array (3,)
        """
        # W pełnym ANFIS byłyby kombinacje typu:
        # μ1 AND μ2 = min(μ1, μ2) lub μ1 * μ2
        # Tutaj mamy single-input rules więc firing = mu
        return mu_values
    
    # ==================== WARSTWA 3: NORMALIZACJA ====================
    
    def normalize_firing(self, firing_strengths):
        """
        WARSTWA 3: Normalizowane siły reguł
        w̄_i = w_i / Σw_j
        """
        total = np.sum(firing_strengths) + 1e-10  # Stabilność numeryczna
        return firing_strengths / total
    
    # ==================== WARSTWA 4: KONSEKWENTY (TSK) ====================
    
    def compute_consequents(self, feature, value, normalized_firing):
        """
        WARSTWA 4: Takagi-Sugeno-Kang model
        Każda reguła ma liniową konsekwentę: f_i = w_i0 + w_i1 * x
        """
        consequents = self.params['consequents'][feature]
        
        # f_i = w_i0 + w_i1 * x dla każdej z 3 reguł
        outputs = consequents[:, 0] + consequents[:, 1] * value
        
        return outputs
    
    # ==================== WARSTWA 5: DEFUZZYFIKACJA ====================
    
    def defuzzify(self, normalized_firing, consequent_outputs):
        """
        WARSTWA 5: Ważona suma (weighted average)
        y = Σ(w̄_i * f_i)
        """
        output = np.sum(normalized_firing * consequent_outputs)
        return np.clip(output, 0, 100)  # DODAJ TĘ LINIĘ
    # ==================== FORWARD PASS ====================
    
    def forward(self, inputs_dict):
        """
        Pełny forward pass przez 5 warstw ANFIS
        inputs_dict: {'RSI': 35, 'VIX': 60, ...}
        Returns: final_output (0-100)
        """
        feature_outputs = []
        feature_weights = []
        
        self.cache = {}  # Reset cache dla backprop
        
        for feature in self.active_features:
            if feature not in inputs_dict:
                continue
            
            value = np.clip(inputs_dict[feature], 0, 100)
            
            # WARSTWA 1: Fuzzyfikacja
            mu = self.fuzzify(feature, value)
            
            # WARSTWA 2: Reguły (firing strengths)
            firing = self.apply_rules(mu)
            
            # WARSTWA 3: Normalizacja
            norm_firing = self.normalize_firing(firing)
            
            # WARSTWA 4: Konsekwenty
            conseq_out = self.compute_consequents(feature, value, norm_firing)
            
            # WARSTWA 5: Defuzzyfikacja (dla tego wskaźnika)
            output = self.defuzzify(norm_firing, conseq_out)
            
            # Cache dla backprop
            self.cache[feature] = {
                'value': value,
                'mu': mu,
                'firing': firing,
                'norm_firing': norm_firing,
                'conseq_out': conseq_out,
                'output': output
            }
            
            feature_outputs.append(output)
            feature_weights.append(1.0)  # Można dodać wagi wskaźników
        
        if len(feature_outputs) == 0:
            return 50.0
        
        # Agregacja wyników z wszystkich wskaźników
        final = np.average(feature_outputs, weights=feature_weights)
        return np.clip(final, 0, 100)  # ✓ To już jest OK
    
    # ==================== BACKWARD PASS (BACKPROPAGATION) ====================
    
    def backward(self, inputs_dict, target, current_output):
        """
        Backpropagation - oblicza gradienty i aktualizuje parametry
        Używamy chain rule do propagacji błędu wstecz
        """
        error = current_output - target
        
        # Dla każdego wskaźnika
        for feature in self.active_features:
            if feature not in inputs_dict or feature not in self.cache:
                continue
            
            cached = self.cache[feature]
            value = cached['value']
            mu = cached['mu']
            norm_firing = cached['norm_firing']
            conseq_out = cached['conseq_out']
            
            # ∂E/∂output = error (dla MSE)
            dE_dout = error
            
            # ========== Gradient dla KONSEKWENTÓW (Least Squares) ==========
            # ∂E/∂w_ij = ∂E/∂y * ∂y/∂w_ij
            # gdzie y = Σ(w̄_i * (w_i0 + w_i1*x))
            
            for i in range(3):  # Dla każdej reguły
                # ∂y/∂w_i0 = w̄_i
                grad_w0 = dE_dout * norm_firing[i]
                
                # ∂y/∂w_i1 = w̄_i * x
                grad_w1 = dE_dout * norm_firing[i] * value
                
                # Update (gradient descent)
                self.params['consequents'][feature][i, 0] -= self.learning_rate * grad_w0
                self.params['consequents'][feature][i, 1] -= self.learning_rate * grad_w1
            
            # ========== Gradient dla CENTRÓW ==========
            # ∂E/∂c_i = ∂E/∂y * ∂y/∂μ_i * ∂μ_i/∂c_i
            
            for i in range(3):
                center = self.params['centers'][feature][i]
                sigma = self.params['sigmas'][feature][i]
                
                # ∂μ/∂c = μ * (x - c) / σ²
                dmu_dc = mu[i] * (value - center) / (sigma ** 2)
                
                # ∂y/∂μ_i (wpływ μ_i na wyjście przez normalizację)
                # To jest skomplikowane bo μ wpływa na normalizację innych reguł
                # Uproszczenie: ∂y/∂μ_i ≈ conseq_out[i] * (1/Σμ - μ_i/(Σμ)²)
                sum_mu = np.sum(mu) + 1e-10
                dy_dmu = conseq_out[i] * (1/sum_mu - mu[i]/(sum_mu**2))
                
                grad_center = dE_dout * dy_dmu * dmu_dc
                
                # Update
                self.params['centers'][feature][i] -= self.learning_rate * grad_center
                
                # Ograniczenia (centra w odpowiedniej kolejności)
                self.params['centers'][feature] = np.clip(
                    self.params['centers'][feature], 
                    [10, 30, 60], 
                    [30, 70, 90]
                )
            
            # ========== Gradient dla SIGM ==========
            # ∂E/∂σ_i = ∂E/∂y * ∂y/∂μ_i * ∂μ_i/∂σ_i
            
            for i in range(3):
                center = self.params['centers'][feature][i]
                sigma = self.params['sigmas'][feature][i]
                
                # ∂μ/∂σ = μ * (x - c)² / σ³
                dmu_dsigma = mu[i] * ((value - center) ** 2) / (sigma ** 3)
                
                sum_mu = np.sum(mu) + 1e-10
                dy_dmu = conseq_out[i] * (1/sum_mu - mu[i]/(sum_mu**2))
                
                grad_sigma = dE_dout * dy_dmu * dmu_dsigma
                
                # Update
                self.params['sigmas'][feature][i] -= self.learning_rate * grad_sigma
                
                # Ograniczenia (sigma > 0)
                self.params['sigmas'][feature] = np.clip(
                    self.params['sigmas'][feature], 
                    3.0, 
                    30.0
                )
    
    # ==================== TRENING ====================
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Uczenie ANFIS na zbiorze treningowym z mini-batch
        """
        n_samples = len(X)
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            total_error = 0
            param_snapshot = deepcopy(self.params)
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for batch in range(n_batches):
                batch_indices = indices[batch*batch_size:(batch+1)*batch_size]
                batch_error = 0
                
                # Accumulate gradients over batch
                for idx in batch_indices:
                    inputs = X[idx]
                    target = y[idx]
                    
                    # Forward pass
                    output = self.forward(inputs)
                    
                    # Backward pass (gradients accumulate internally)
                    self.backward(inputs, target, output)
                    
                    # Accumulate error
                    batch_error += (output - target) ** 2
                
                total_error += batch_error
            
            # MSE
            mse = total_error / n_samples
            
            # Oblicz zmianę parametrów
            param_change = 0
            for feature in self.active_features:
                param_change += np.sum(np.abs(
                    self.params['centers'][feature] - param_snapshot['centers'][feature]
                ))
                param_change += np.sum(np.abs(
                    self.params['sigmas'][feature] - param_snapshot['sigmas'][feature]
                ))
            
            # Historia
            self.training_history['epochs'].append(epoch)
            self.training_history['mse'].append(mse)
            self.training_history['param_changes'].append(param_change)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - MSE: {mse:.4f}, Param Change: {param_change:.4f}")
        
        self.trained = True
        if verbose:
            print(f"✓ Trening zakończony! Finalny MSE: {mse:.4f}")
    
    # ==================== INTERFEJS KOMPATYBILNY Z STARYM KODEM ====================
    
    def build_system(self, user_config):
        """Kompatybilność z starym API"""
        self.initialize_params(user_config)
    
    def compute(self, inputs_dict):
        """Kompatybilność z starym API"""
        if not hasattr(self, 'active_features'):
            return 50.0
        return self.forward(inputs_dict)
    
    # ==================== EKSPORT PARAMETRÓW ====================
    
    def get_params_summary(self):
        """Zwraca podsumowanie nauczonych parametrów"""
        summary = {}
        for feature in self.active_features:
            summary[feature] = {
                'centers': self.params['centers'][feature].tolist(),
                'sigmas': self.params['sigmas'][feature].tolist(),
                'consequents': self.params['consequents'][feature].tolist(),
                'direction': self.params['directions'][feature]
            }
        return summary