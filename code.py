"""
Project: Automated Rediscovery of Asymptotic Number Theory
Author: Mohamed Dhia Eddine Thabet
Description: Benchmarking Deep Learning vs. Symbolic Regression for Prime Number prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
from scipy.special import expi
from scipy.optimize import fsolve

# ==========================================
# GLOBAL SETTINGS
# ==========================================
TRAINING_LIMIT = 10000   # Train on first 10k primes
TEST_LIMIT = 50000       # Validate on next 40k
BENCHMARK_TARGET = 1000000 # Test extrapolation on 1 Millionth prime

# ==========================================
# PART 1: DATA GENERATION UTILS
# ==========================================
def get_primes(limit):
    """Generates primes using Sieve of Eratosthenes."""
    print(f"[Data] Generating primes up to limit index {limit}...")
    # Approximate upper bound for nth prime is n(ln n + ln ln n)
    upper_bound = int(limit * (math.log(limit) + math.log(math.log(limit)))) + 100
    
    sieve = np.ones(upper_bound // 2, dtype=bool)
    sieve[0] = False
    for i in range(3, int(upper_bound**0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i*i // 2 :: i] = False
            
    primes = [2] + [2*i + 1 for i in range(1, upper_bound // 2) if sieve[i]]
    return np.array(primes[:limit])

# ==========================================
# PART 2: DEEP LEARNING EXPERIMENT (BASELINE)
# ==========================================
def run_deep_learning_experiment(primes):
    print("\n" + "="*40)
    print("   EXPERIMENT 1: DEEP LEARNING (TRANSFORMER)")
    print("="*40)
    
    # 1. Data Prep: Predict Gaps
    gaps = np.diff(primes)
    SEQ_LEN = 32
    X, y = [], []
    for i in range(len(gaps) - SEQ_LEN):
        X.append(gaps[i : i + SEQ_LEN])
        y.append(gaps[i + SEQ_LEN])
    
    X = np.array(X).reshape(-1, SEQ_LEN, 1)
    y = np.array(y)
    
    # Normalize log-scale to help neural net
    X_log = np.log1p(X)
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, shuffle=False)
    
    # 2. Model: Simple Transformer
    inputs = layers.Input(shape=(SEQ_LEN, 1))
    x = layers.MultiHeadAttention(key_dim=32, num_heads=2, dropout=0.1)(inputs, inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    
    print("[DL] Training Transformer...")
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.2)
    
    # 3. Evaluate
    print("[DL] Evaluating...")
    preds_log = model.predict(X_test)
    preds_real = np.expm1(preds_log).flatten()
    actual_real = np.expm1(y_test).flatten()
    
    mae = np.mean(np.abs(preds_real - actual_real))
    print(f"[DL RESULT] Mean Absolute Error on Gaps: {mae:.4f}")
    print("Observation: The model likely converged to the mean gap size (Probabilistic Collapse).")

# ==========================================
# PART 3: SYMBOLIC REGRESSION (THE WINNER)
# ==========================================
def run_symbolic_regression_experiment(primes):
    print("\n" + "="*40)
    print("   EXPERIMENT 2: SYMBOLIC REGRESSION")
    print("="*40)
    
    # Data: Input is Index (n), Output is Prime Value (p_n)
    n_train = len(primes)
    X = np.arange(1, n_train + 1).reshape(-1, 1)
    y = primes
    
    print("[Symbolic] Evolving formulas (Genetic Programming)...")
    est = SymbolicRegressor(
        population_size=2000,
        generations=15,
        stopping_criteria=0.01,
        p_crossover=0.7, 
        p_subtree_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'sin', 'cos'),
        random_state=42
    )
    
    est.fit(X, y)
    
    print("\n[Symbolic RESULT] Best Formula Found:")
    print(est._program)
    return est

# ==========================================
# PART 4: BENCHMARK & VISUALIZATION
# ==========================================
def run_final_benchmark(symbolic_model):
    print("\n" + "="*40)
    print(f"   FINAL BENCHMARK: THE {BENCHMARK_TARGET:,}th PRIME")
    print("="*40)
    
    # 1. Get Real Truth (Standard Math Library)
    # We use Inverse Logarithmic Integral as Gold Standard Approximation
    def inverse_li(n):
        func = lambda x: expi(np.log(x)) - n
        return fsolve(func, n * np.log(n))[0]

    # The known 1,000,000th prime is 15,485,863
    # For automation, we calculate approximation or hardcode the truth if known
    real_prime = 15485863 # Hardcoded truth for this specific test case
    
    # 2. Get Predictions
    # A. Basic Formula (n ln n)
    basic_pred = BENCHMARK_TARGET * np.log(BENCHMARK_TARGET)
    
    # B. AI Prediction
    ai_pred = symbolic_model.predict(np.array([[BENCHMARK_TARGET]]))[0]
    
    # C. Gold Standard Math (Li^-1)
    math_pred = inverse_li(BENCHMARK_TARGET)
    
    # 3. Calculate Errors
    basic_error = abs(real_prime - basic_pred)
    ai_error = abs(real_prime - ai_pred)
    math_error = abs(real_prime - math_pred)
    
    print(f"Target: {real_prime}")
    print(f"Basic Formula Error: {basic_error:,.0f}")
    print(f"AI Model Error:      {ai_error:,.0f}")
    print(f"Gold Standard Error: {math_error:,.0f}")
    
    # 4. Plotting
    categories = ['Basic Formula\n(n ln n)', 'Your AI Model', 'Gold Standard\n(Inverse Li)']
    errors = [basic_error, ai_error, math_error]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, errors, color=['gray', 'red', 'blue'], alpha=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval):,}', 
                 va='bottom', ha='center', fontweight='bold')
    
    plt.yscale('log')
    plt.ylabel('Absolute Error (Log Scale)')
    plt.title(f'Error Benchmark at n={BENCHMARK_TARGET:,}')
    plt.savefig('benchmark_graph.png')
    print("[Graph] Saved benchmark_graph.png")
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Generate Data
    dataset_primes = get_primes(TRAINING_LIMIT)
    
    # 2. Run Deep Learning (To show it fails/plateaus)
    run_deep_learning_experiment(dataset_primes)
    
    # 3. Run Symbolic Regression (To show it succeeds)
    ai_model = run_symbolic_regression_experiment(dataset_primes)
    
    # 4. Prove it on the 1 Millionth Prime
    run_final_benchmark(ai_model)
