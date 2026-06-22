# ── Deep Learning Model ───────────────────────────────────────
# LSTM + GRU for price forecasting
# Predicts next 30 days of closing price

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

import sys
sys.path.append("..")
from config import SEQ_LENGTH, PREDICT_DAYS, TEST_SIZE, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)

# ── TensorFlow import with error handling ─────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout,
        Bidirectional, BatchNormalization
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau
    )
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not available — DL model disabled")


# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_sequences(
    df: pd.DataFrame,
    seq_length: int = SEQ_LENGTH
) -> tuple:
    """
    Converts OHLCV data into sequences for LSTM.

    Input:  raw OHLCV DataFrame
    Output: X (sequences), y (next close), scaler
    """
    # Use multiple features for better prediction
    features = ["close", "high", "low", "volume"]
    features = [f for f in features if f in df.columns]

    data   = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    close_idx = features.index("close")

    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length: i])         # Past 60 days
        y.append(scaled[i, close_idx])               # Next close price

    X = np.array(X)
    y = np.array(y)

    # Train / test split (no shuffle — time series!)
    split    = int(len(X) * (1 - TEST_SIZE))
    X_train  = X[:split]
    X_test   = X[split:]
    y_train  = y[:split]
    y_test   = y[split:]

    return X_train, X_test, y_train, y_test, scaler, features


def prepare_forecast_input(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    features: list,
    seq_length: int = SEQ_LENGTH
) -> np.ndarray:
    """Prepares last seq_length rows for forecasting."""
    data   = df[features].values[-seq_length:]
    scaled = scaler.transform(data)
    return scaled.reshape(1, seq_length, len(features))


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════

def build_lstm(input_shape: tuple) -> "Sequential":
    """
    Stacked LSTM model.
    Good at learning long-term price patterns.
    """
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(32, return_sequences=False),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1),
    ], name="LSTM_Model")

    model.compile(
        optimizer = Adam(learning_rate=0.001),
        loss      = "huber",       # Robust to outliers
        metrics   = ["mae"],
    )
    return model


def build_gru(input_shape: tuple) -> "Sequential":
    """
    GRU model — faster than LSTM, similar accuracy.
    """
    model = Sequential([
        GRU(128, return_sequences=True,
            input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),

        GRU(64, return_sequences=False),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="GRU_Model")

    model.compile(
        optimizer = Adam(learning_rate=0.001),
        loss      = "huber",
        metrics   = ["mae"],
    )
    return model


def build_bilstm(input_shape: tuple) -> "Sequential":
    """
    Bidirectional LSTM — reads sequence forward AND backward.
    Best accuracy but slowest to train.
    """
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=input_shape
        ),
        Dropout(0.2),
        BatchNormalization(),

        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1),
    ], name="BiLSTM_Model")

    model.compile(
        optimizer = Adam(learning_rate=0.001),
        loss      = "huber",
        metrics   = ["mae"],
    )
    return model


# ═══════════════════════════════════════════════════════════════
# MAIN DL MODEL CLASS
# ═══════════════════════════════════════════════════════════════

class DLStockModel:
    """
    Trains LSTM + GRU + BiLSTM ensemble.
    Forecasts next PREDICT_DAYS closing prices.
    """

    def __init__(self, ticker: str, model_type: str = "lstm"):
        self.ticker     = ticker
        self.model_type = model_type   # lstm / gru / bilstm
        self.model      = None
        self.scaler     = None
        self.features   = []
        self.trained    = False
        self.metrics    = {}

    # ── Train ─────────────────────────────────────────────────
    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> dict:
        """Train the DL model on historical price data."""

        if not TF_AVAILABLE:
            return {"error": "TensorFlow not installed"}

        if len(df) < SEQ_LENGTH + 50:
            return {"error": f"Need at least {SEQ_LENGTH + 50} days of data"}

        # Prepare sequences
        X_train, X_test, y_train, y_test, self.scaler, self.features = \
            prepare_sequences(df, SEQ_LENGTH)

        input_shape = (X_train.shape[1], X_train.shape[2])

        # Build model based on type
        if self.model_type == "gru":
            self.model = build_gru(input_shape)
        elif self.model_type == "bilstm":
            self.model = build_bilstm(input_shape)
        else:
            self.model = build_lstm(input_shape)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor   = "val_loss",
                patience  = 10,
                restore_best_weights = True,
                verbose   = 0,
            ),
            ReduceLROnPlateau(
                monitor  = "val_loss",
                factor   = 0.5,
                patience = 5,
                verbose  = 0,
            ),
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs          = epochs,
            batch_size      = batch_size,
            validation_data = (X_test, y_test),
            callbacks       = callbacks,
            verbose         = 1,
        )

        # Evaluate
        y_pred   = self.model.predict(X_test, verbose=0)

        # Inverse transform for real prices
        close_idx = self.features.index("close")
        n_feat    = len(self.features)

        def inverse_close(arr):
            dummy = np.zeros((len(arr), n_feat))
            dummy[:, close_idx] = arr.flatten()
            return self.scaler.inverse_transform(dummy)[:, close_idx]

        y_test_real = inverse_close(y_test)
        y_pred_real = inverse_close(y_pred)

        mape = mean_absolute_percentage_error(y_test_real, y_pred_real)

        self.trained = True
        self.metrics = {
            "model_type":    self.model_type,
            "mape":          round(mape * 100, 2),
            "accuracy_pct":  round((1 - mape) * 100, 2),
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
            "epochs_run":    len(history.history["loss"]),
            "final_loss":    round(history.history["val_loss"][-1], 6),
        }

        self._save()
        return self.metrics

    # ── Forecast ──────────────────────────────────────────────
    def forecast(
        self,
        df: pd.DataFrame,
        days: int = PREDICT_DAYS,
    ) -> dict:
        """
        Forecast next `days` closing prices.
        Uses recursive prediction (each prediction feeds next input).
        """
        if not self.trained:
            loaded = self._load()
            if not loaded:
                return {"error": "Model not trained"}

        if not TF_AVAILABLE:
            return {"error": "TensorFlow not available"}

        close_idx = self.features.index("close")
        n_feat    = len(self.features)

        # Start with last SEQ_LENGTH rows
        data      = df[self.features].values
        scaled    = self.scaler.transform(data)
        sequence  = scaled[-SEQ_LENGTH:].copy()

        predictions = []

        for _ in range(days):
            inp   = sequence.reshape(1, SEQ_LENGTH, n_feat)
            pred  = self.model.predict(inp, verbose=0)[0, 0]

            # Build next row (shift everything, update close)
            next_row              = sequence[-1].copy()
            next_row[close_idx]   = pred
            sequence              = np.vstack([sequence[1:], next_row])

            predictions.append(pred)

        # Inverse transform predictions to real prices
        dummy = np.zeros((len(predictions), n_feat))
        dummy[:, close_idx] = predictions
        real_prices = self.scaler.inverse_transform(dummy)[:, close_idx]

        # Generate future dates
        last_date  = df.index[-1]
        future_dates = pd.bdate_range(
            start = last_date + pd.Timedelta(days=1),
            periods = days
        )

        current_price = float(df["close"].iloc[-1])
        final_price   = float(real_prices[-1])
        change_pct    = (final_price - current_price) / current_price * 100

        return {
            "ticker":          self.ticker,
            "current_price":   round(current_price, 2),
            "forecast_prices": [round(p, 2) for p in real_prices],
            "forecast_dates":  [str(d.date()) for d in future_dates],
            "final_price":     round(final_price, 2),
            "change_pct":      round(change_pct, 2),
            "direction":       "📈 UP" if change_pct > 0 else "📉 DOWN",
            "model_type":      self.model_type,
            "accuracy_pct":    self.metrics.get("accuracy_pct", 0),
        }

    # ── Save / Load ───────────────────────────────────────────
    def _save(self):
        base = os.path.join(
            MODEL_DIR,
            f"{self.ticker.replace('.','_')}_{self.model_type}"
        )
        # Save Keras model
        self.model.save(base + ".keras")
        # Save scaler + metadata
        with open(base + "_meta.pkl", "wb") as f:
            pickle.dump({
                "scaler":   self.scaler,
                "features": self.features,
                "metrics":  self.metrics,
            }, f)

    def _load(self) -> bool:
        base = os.path.join(
            MODEL_DIR,
            f"{self.ticker.replace('.','_')}_{self.model_type}"
        )
        if not os.path.exists(base + ".keras"):
            return False
        self.model = load_model(base + ".keras")
        with open(base + "_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.scaler   = meta["scaler"]
        self.features = meta["features"]
        self.metrics  = meta["metrics"]
        self.trained  = True
        return True


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE FORECAST
# ═══════════════════════════════════════════════════════════════

def ensemble_forecast(ticker: str, df: pd.DataFrame) -> dict:
    """
    Trains LSTM + GRU, averages their forecasts.
    More reliable than single model.
    """
    results = {}

    for model_type in ["lstm", "gru"]:
        print(f"\nTraining {model_type.upper()} for {ticker}...")
        model = DLStockModel(ticker, model_type)

        # Try loading first
        if not model._load():
            metrics = model.train(df, epochs=50)
            if "error" in metrics:
                print(f"[WARN] {model_type} failed: {metrics['error']}")
                continue

        forecast = model.forecast(df)
        if "error" not in forecast:
            results[model_type] = forecast

    if not results:
        return {"error": "All models failed"}

    # Average the forecasts
    all_prices = np.array([
        r["forecast_prices"] for r in results.values()
    ])
    avg_prices    = np.mean(all_prices, axis=0)

    current       = list(results.values())[0]["current_price"]
    final         = float(avg_prices[-1])
    change_pct    = (final - current) / current * 100
    dates         = list(results.values())[0]["forecast_dates"]

    return {
        "ticker":          ticker,
        "current_price":   current,
        "forecast_prices": [round(p, 2) for p in avg_prices],
        "forecast_dates":  dates,
        "final_price":     round(final, 2),
        "change_pct":      round(change_pct, 2),
        "direction":       "📈 UP" if change_pct > 0 else "📉 DOWN",
        "models_used":     list(results.keys()),
        "individual":      results,
    }