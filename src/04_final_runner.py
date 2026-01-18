import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from regime import detect_regimes
from strategy import apply_strategy
from backtest import run_backtest
from ml_models import train_ml_models
from analysis import perform_outlier_analysis, generate_plots

def main():
    print("=== STARTING FINAL QUANT PIPELINE ===")
    
    # 1. Load Data
    print("\n[1/5] Loading Feature Data...")
    df = pd.read_csv('data/nifty_features_5min.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Detect Regimes
    print("\n[2/5] Detect Regimes...")
    df = detect_regimes(df)
    
    # 3. Train ML Models
    print("\n[3/5] Training ML Models (XGBoost & LSTM)...")
    df, xgb_model, lstm_model = train_ml_models(df)
    
    # 4. Strategy & Backtest
    print("\n[4/5] Running Strategy...")
    # Baseline Strategy
    df_base = apply_strategy(df.copy())
    results_base, _ = run_backtest(df_base)
    
    print(f"Baseline Return: {results_base['Total Return (%)']}%")
    
    # 5. Analysis
    print("\n[5/5] Analysis & Plots...")
    perform_outlier_analysis(df_base)
    generate_plots(df_base)
    
    print("\n=== PIPELINE COMPLETE ===")
    print("Check 'results/' for CSVs and 'plots/' for images.")

if __name__ == "__main__":
    main()