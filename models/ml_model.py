# ── ML Model ──────────────────────────────────────────────────
# Random Forest + XGBoost for Buy/Sell/Hold classification
# Uses 30+ technical indicators as features

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import sys
sys.path.append("..")
from config import (
    N_ESTIMATORS, RANDOM_STATE, TEST_SIZE,
    PREDICT_DAYS, MODEL_DIR
)
from utils.indicators import compute_all, ML_FEATURES

os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# LABEL CREATION
# ═══════════════════════════════════════════════════════════════

def create_labels(df: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """
    Creates Buy/Sell/Hold labels based on future returns.

    Logic:
      future return > +3%  → BUY  (2)
      future return < -3%  → SELL (0)
      else                 → HOLD (1)

    horizon = how many days ahead to look
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1

    labels = pd.Series(1, index=df.index)   # Default = HOLD
    labels[future_return >  0.03] = 2       # BUY
    labels[future_return < -0.03] = 0       # SELL

    return labels


# ═══════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Adds all indicators, creates labels, returns X and y.
    """
    # Add all technical indicators
    df = compute_all(df)

    # Create labels (10-day horizon)
    labels = create_labels(df, horizon=10)

    # Use only available ML features
    available = [f for f in ML_FEATURES if f in df.columns]
    X = df[available].copy()
    y = labels

    # Align and drop NaN rows
    combined = pd.concat([X, y.rename("label")], axis=1).dropna()
    # Drop last 10 rows (no future data for labels)
    combined = combined.iloc[:-10]

    X = combined[available]
    y = combined["label"].astype(int)

    return X, y, available


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

class MLStockModel:
    """
    Trains Random Forest + XGBoost on a stock's OHLCV history.
    Combines both into an ensemble for final signal.
    """

    def __init__(self, ticker: str):
        self.ticker   = ticker
        self.rf       = None
        self.xgb      = None
        self.scaler   = StandardScaler()
        self.features = []
        self.trained  = False
        self.metrics  = {}

    # ── Train ─────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train both models on historical OHLCV data.
        Returns accuracy metrics.
        """
        if len(df) < 200:
            return {"error": "Need at least 200 days of data"}

        X, y, self.features = prepare_features(df)

        if len(X) < 100:
            return {"error": "Not enough clean data after indicators"}

        # Train / Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            shuffle=False     # Time series — no shuffling!
        )

        # Scale features
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        # ── Random Forest ──────────────────────────────────────
        self.rf = RandomForestClassifier(
            n_estimators = N_ESTIMATORS,
            max_depth    = 10,
            min_samples_split = 5,
            class_weight = "balanced",
            random_state = RANDOM_STATE,
            n_jobs       = -1,
        )
        self.rf.fit(X_train_sc, y_train)
        rf_pred = self.rf.predict(X_test_sc)
        rf_acc  = accuracy_score(y_test, rf_pred)

        # ── XGBoost ────────────────────────────────────────────
        self.xgb = XGBClassifier(
            n_estimators      = N_ESTIMATORS,
            max_depth         = 6,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            use_label_encoder = False,
            eval_metric       = "mlogloss",
            random_state      = RANDOM_STATE,
            verbosity         = 0,
        )
        self.xgb.fit(X_train_sc, y_train)
        xgb_pred = self.xgb.predict(X_test_sc)
        xgb_acc  = accuracy_score(y_test, xgb_pred)

        self.trained = True

        # ── Feature Importance (top 10) ────────────────────────
        importance = pd.Series(
            self.rf.feature_importances_,
            index=self.features
        ).sort_values(ascending=False).head(10)

        self.metrics = {
            "rf_accuracy":  round(rf_acc * 100, 1),
            "xgb_accuracy": round(xgb_acc * 100, 1),
            "train_size":   len(X_train),
            "test_size":    len(X_test),
            "top_features": importance.to_dict(),
            "label_dist": {
                "BUY":  int((y == 2).sum()),
                "HOLD": int((y == 1).sum()),
                "SELL": int((y == 0).sum()),
            }
        }

        # Save models
        self._save()

        return self.metrics

    # ── Predict ───────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict signal for the latest candle.
        Returns signal + confidence score.
        """
        if not self.trained:
            loaded = self._load()
            if not loaded:
                return {"error": "Model not trained yet"}

        # Compute indicators on full df
        df_ind = compute_all(df)
        available = [f for f in self.features if f in df_ind.columns]

        if df_ind.empty or len(df_ind) < 5:
            return {"error": "Insufficient data"}

        # Use last row for prediction
        X_latest = df_ind[available].iloc[-1:].fillna(0)
        X_scaled = self.scaler.transform(X_latest)

        # RF prediction + probabilities
        rf_proba  = self.rf.predict_proba(X_scaled)[0]
        rf_signal = self.rf.predict(X_scaled)[0]

        # XGB prediction + probabilities
        xgb_proba  = self.xgb.predict_proba(X_scaled)[0]
        xgb_signal = self.xgb.predict(X_scaled)[0]

        # Ensemble: average probabilities
        ensemble_proba  = (rf_proba + xgb_proba) / 2
        ensemble_signal = int(np.argmax(ensemble_proba))

        # Map to labels
        label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

        # Confidence = max probability
        confidence = float(np.max(ensemble_proba)) * 100

        return {
            "signal":        label_map[ensemble_signal],
            "confidence":    round(confidence, 1),
            "probabilities": {
                "SELL": round(float(ensemble_proba[0]) * 100, 1),
                "HOLD": round(float(ensemble_proba[1]) * 100, 1),
                "BUY":  round(float(ensemble_proba[2]) * 100, 1),
            },
            "rf_signal":     label_map[rf_signal],
            "xgb_signal":    label_map[xgb_signal],
            "agreement":     rf_signal == xgb_signal,
        }

    # ── Feature Importance ────────────────────────────────────
    def get_feature_importance(self) -> dict:
        """Returns top features driving predictions."""
        if not self.trained or not self.rf:
            return {}
        return pd.Series(
            self.rf.feature_importances_,
            index=self.features
        ).sort_values(ascending=False).head(15).to_dict()

    # ── Save / Load ───────────────────────────────────────────
    def _save(self):
        path = os.path.join(
            MODEL_DIR, f"{self.ticker.replace('.','_')}_ml.pkl"
        )
        with open(path, "wb") as f:
            pickle.dump({
                "rf":       self.rf,
                "xgb":      self.xgb,
                "scaler":   self.scaler,
                "features": self.features,
                "metrics":  self.metrics,
            }, f)

    def _load(self) -> bool:
        path = os.path.join(
            MODEL_DIR, f"{self.ticker.replace('.','_')}_ml.pkl"
        )
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.rf       = data["rf"]
        self.xgb      = data["xgb"]
        self.scaler   = data["scaler"]
        self.features = data["features"]
        self.metrics  = data["metrics"]
        self.trained  = True
        return True


# ═══════════════════════════════════════════════════════════════
# QUICK TRAIN + PREDICT PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_ml_pipeline(ticker: str, df: pd.DataFrame) -> dict:
    """
    One-call function: trains model and returns prediction.
    Used by the screener to process each stock.
    """
    model   = MLStockModel(ticker)

    # Try loading existing model first
    loaded  = model._load()

    if not loaded:
        print(f"Training ML model for {ticker}...")
        metrics = model.train(df)
        if "error" in metrics:
            return {"ticker": ticker, "error": metrics["error"]}

    prediction = model.predict(df)
    prediction["ticker"]  = ticker
    prediction["metrics"] = model.metrics

    return prediction