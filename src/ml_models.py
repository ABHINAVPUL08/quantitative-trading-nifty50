import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os


def prepare_targets(df):
    """
    Binary target:
    1 if price rises in next 5 candles, else 0
    """
    # Detect price column safely
    if "close_spot" in df.columns:
        price_col = "close_spot"
    elif "close" in df.columns:
        price_col = "close"
    else:
        raise ValueError(f"No price column found. Columns: {list(df.columns)}")

    df["future_return"] = df[price_col].shift(-5) - df[price_col]
    df["target"] = (df["future_return"] > 0).astype(int)
    return df.dropna()


def train_ml_models():
    print("Running ML Models pipeline...")

    # Load data
    df = pd.read_csv("data/nifty_features_5min.csv")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    # Prepare target
    df = prepare_targets(df)

    # ---- Candidate features (we'll auto-filter) ----
    CANDIDATE_FEATURES = [
        "returns",
        "ema_fast", "ema_slow",
        "ema_5", "ema_15",
        "iv", "vega", "gamma", "delta",
        "pcr", "basis",
        "regime", "regime_label"
    ]

    # Keep only features that exist
    FEATURES = [f for f in CANDIDATE_FEATURES if f in df.columns]

    if len(FEATURES) == 0:
        raise ValueError("No valid ML features found in dataset")

    print("\nUsing ML features:")
    print(FEATURES)

    X = df[FEATURES]
    y = df["target"]

    # Time-aware split
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nXGBoost Accuracy: {acc:.2%}")

    # Save predictions
    df["ml_signal"] = model.predict(X)
    df["ml_prob"] = model.predict_proba(X)[:, 1]

    os.makedirs("models", exist_ok=True)
    df[["timestamp", "ml_signal", "ml_prob"]].to_csv(
        "models/ml_predictions.csv",
        index=False
    )

    print("\nML predictions saved to models/ml_predictions.csv ✅")


if __name__ == "__main__":
    train_ml_models()
