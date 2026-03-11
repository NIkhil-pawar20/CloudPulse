"""
predictor.py — ML Pipeline
============================
CloudOptimizer AI  |  Layer 2: Intelligence Engine

Based on: Osypanka & Nawrocki (2022)
"Resource Usage Cost Optimization in Cloud Computing Using Machine Learning"

Feature Engineering  : 13 features (cyclical encodings, lag, rolling stats)
Models evaluated     : BayesianRidge, RandomForest, GradientBoosting, MLP
Model selection      : Lowest RMSE on hold-out (last 168 hours)
Forecast horizon     : 168 hours (7 days)

Usage:
    pred = CloudPredictor(db)
    result = pred.run(component="paas_payment", resource="acu")
"""

import uuid
import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FORECAST_HORIZON = 168          # hours to predict
HOLDOUT_HOURS    = 168          # validation window (last 168 h of training set)
MIN_TRAIN_ROWS   = 200          # minimum cleaned rows needed to train

MODELS = {
    "BayesianRidge": BayesianRidge(),
    "RandomForest":  RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42
    ),
    "MLPRegressor": MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO/SQLite datetime strings to datetime (UTC-naive)."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str!r}")


def engineer_features(
    timestamps: list[str],
    values: list[float],
) -> tuple[np.ndarray, list[str]]:
    """
    Build the 13-feature design matrix.

    Features
    --------
    1.  hour_sin       — sin encoding of hour-of-day  (daily cycle)
    2.  hour_cos       — cos encoding of hour-of-day
    3.  day_sin        — sin encoding of day-of-week  (weekly cycle)
    4.  day_cos        — cos encoding of day-of-week
    5.  is_weekend     — 1 if Sat/Sun else 0
    6.  lag_1          — value 1 h ago
    7.  lag_2          — value 2 h ago
    8.  lag_24         — value 24 h ago (same hour yesterday)
    9.  lag_168        — value 168 h ago (same hour last week)
    10. rolling_mean_6 — rolling mean over last 6 h
    11. rolling_std_6  — rolling std  over last 6 h
    12. rolling_mean_24— rolling mean over last 24 h
    13. base_value     — raw normalised value (for tree models)

    Returns
    -------
    X : np.ndarray  shape (n_valid_rows, 13)
    feature_names : list[str]
    """
    feature_names = [
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "is_weekend",
        "lag_1", "lag_2", "lag_24", "lag_168",
        "rolling_mean_6", "rolling_std_6", "rolling_mean_24",
        "base_value",
    ]
    assert len(feature_names) == 13, "Feature count must be exactly 13"

    dts = [_parse_ts(ts) for ts in timestamps]
    n = len(values)
    rows = []

    for i in range(n):
        dt = dts[i]

        # ── Cyclical encodings ──────────────────────────────────────────────
        h = dt.hour
        d = dt.weekday()  # 0=Mon … 6=Sun
        hour_sin = math.sin(2 * math.pi * h / 24)
        hour_cos = math.cos(2 * math.pi * h / 24)
        day_sin  = math.sin(2 * math.pi * d / 7)
        day_cos  = math.cos(2 * math.pi * d / 7)
        is_weekend = 1 if d >= 5 else 0

        # ── Lag features (0 if index out of range) ─────────────────────────
        lag_1   = values[i - 1]   if i >= 1   else 0.0
        lag_2   = values[i - 2]   if i >= 2   else 0.0
        lag_24  = values[i - 24]  if i >= 24  else 0.0
        lag_168 = values[i - 168] if i >= 168 else 0.0

        # ── Rolling stats ──────────────────────────────────────────────────
        win6  = values[max(0, i - 6):i]   if i > 0 else [values[i]]
        win24 = values[max(0, i - 24):i]  if i > 0 else [values[i]]

        rolling_mean_6  = statistics.mean(win6)  if win6  else values[i]
        rolling_std_6   = statistics.stdev(win6) if len(win6) >= 2 else 0.0
        rolling_mean_24 = statistics.mean(win24) if win24 else values[i]

        row = [
            hour_sin, hour_cos, day_sin, day_cos,
            is_weekend,
            lag_1, lag_2, lag_24, lag_168,
            rolling_mean_6, rolling_std_6, rolling_mean_24,
            values[i],
        ]
        rows.append(row)

    X = np.array(rows, dtype=np.float64)
    return X, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CloudPredictor:
    """
    Multi-model ML predictor with automatic best-model selection.

    Parameters
    ----------
    db : CloudOptimizerDB | None
        Pass None for testing without a database.
    """

    def __init__(self, db=None):
        self.db = db
        self._trained_models: dict = {}
        self._scaler: Optional[StandardScaler] = None
        self._best_model_name: Optional[str] = None
        self._rmse_table: dict = {}

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def run(
        self,
        component: str,
        resource: str,
        cleaned_series: Optional[list] = None,
    ) -> dict:
        """
        Train all models, pick the best by RMSE, forecast 168 hours.

        Parameters
        ----------
        cleaned_series : list of dicts or list of floats (optional)
            If supplied, used directly.  Otherwise fetched from self.db.

        Returns
        -------
        dict with keys:
            best_model      str
            rmse_table      dict[model_name -> float]
            forecast        list[dict]  (target_timestamp, predicted_value)
            run_id          str
        """
        # ── 1. Load data ──────────────────────────────────────────────────────
        timestamps, values = self._load_cleaned(component, resource, cleaned_series)

        if len(values) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Need ≥{MIN_TRAIN_ROWS} cleaned rows to train "
                f"(got {len(values)} for {component}/{resource})."
            )

        # ── 2. Feature matrix ────────────────────────────────────────────────
        X, feature_names = engineer_features(timestamps, values)
        y = np.array(values, dtype=np.float64)

        # ── 3. Train / validation split ───────────────────────────────────────
        split = max(HOLDOUT_HOURS, len(values) - HOLDOUT_HOURS)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # ── 4. Scale features ─────────────────────────────────────────────────
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)
        self._scaler = scaler

        # ── 5. Train all models & evaluate ───────────────────────────────────
        rmse_table: dict[str, float] = {}
        trained: dict = {}

        for name, model in MODELS.items():
            try:
                m = model.__class__(**model.get_params())  # fresh instance
                m.fit(X_train_sc, y_train)
                if len(X_val_sc) > 0:
                    y_pred = m.predict(X_val_sc)
                    rmse = math.sqrt(mean_squared_error(y_val, y_pred))
                else:
                    # No hold-out; retrain on full set, report training RMSE
                    y_pred = m.predict(X_train_sc)
                    rmse = math.sqrt(mean_squared_error(y_train, y_pred))
                rmse_table[name] = round(rmse, 4)
                trained[name] = m
            except Exception as exc:
                print(f"  [predictor] Model {name} failed: {exc}")
                rmse_table[name] = float("inf")

        # ── 6. Pick best model ────────────────────────────────────────────────
        best_name = min(rmse_table, key=rmse_table.get)
        self._best_model_name = best_name
        self._trained_models = trained
        self._rmse_table = rmse_table

        print(
            f"  [predictor] Best model → {best_name}  "
            f"(RMSE={rmse_table[best_name]:.4f})"
        )

        # ── 7. Generate 168-hour forecast using best model ────────────────────
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        forecast = self._forecast(
            component, resource, timestamps, values, best_name, trained, scaler, run_id
        )

        # ── 8. Persist to DB ──────────────────────────────────────────────────
        if self.db is not None:
            self._save_to_db(component, resource, forecast, rmse_table, best_name, run_id)

        return {
            "run_id":     run_id,
            "best_model": best_name,
            "rmse_table": rmse_table,
            "forecast":   forecast,
            "feature_names": feature_names,
        }

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _forecast(
        self,
        component, resource, timestamps, values,
        best_name, trained, scaler, run_id,
    ) -> list[dict]:
        """
        Generate FORECAST_HORIZON predictions auto-regressively.

        The predictor appends its own previous outputs to simulate future
        lag features (lag_1, lag_2, lag_24, lag_168).
        """
        last_ts = _parse_ts(timestamps[-1])
        future_timestamps = [
            (last_ts + timedelta(hours=h + 1)).strftime("%Y-%m-%d %H:%M:%S")
            for h in range(FORECAST_HORIZON)
        ]

        # Extend the running series with the historical tail (for lag features)
        running_values = list(values)
        running_ts     = list(timestamps)

        forecast = []
        model = trained[best_name]

        for i, fts in enumerate(future_timestamps):
            running_ts.append(fts)
            running_values.append(running_values[-1])  # provisional placeholder

            X_all, _ = engineer_features(running_ts, running_values)
            x_new = X_all[-1].reshape(1, -1)
            x_new_sc = scaler.transform(x_new)

            pred = float(model.predict(x_new_sc)[0])
            pred = max(0.0, pred)  # clamp to non-negative

            running_values[-1] = pred  # replace placeholder with actual forecast

            forecast.append(
                {
                    "target_timestamp": fts,
                    "predicted_value":  round(pred, 4),
                }
            )

        return forecast

    def _save_to_db(self, component, resource, forecast, rmse_table, best_name, run_id):
        """Persist all model predictions (marking is_best=1 for winner)."""
        records = []
        for entry in forecast:
            for model_name, rmse in rmse_table.items():
                records.append(
                    {
                        "run_id":           run_id,
                        "component":        component,
                        "resource":         resource,
                        "target_timestamp": entry["target_timestamp"],
                        "predicted_value":  entry["predicted_value"],
                        "model_name":       model_name,
                        "is_best":          1 if model_name == best_name else 0,
                    }
                )
        # Only best-model forecasts have meaningful predicted_value rows.
        # Non-best models share the same predicted_value here for DB completeness;
        # the is_best flag distinguishes which to trust.
        self.db.insert_predictions(records)

    def _load_cleaned(self, component, resource, cleaned_series):
        """Return (timestamps: list[str], values: list[float])."""
        if cleaned_series is not None:
            if cleaned_series and isinstance(cleaned_series[0], dict):
                timestamps = [r.get("timestamp", r.get("target_timestamp", "")) for r in cleaned_series]
                values     = [float(r.get("cleaned_value", r.get("value", 0))) for r in cleaned_series]
            else:
                n = len(cleaned_series)
                timestamps = [f"2024-01-01 {i % 24:02d}:00:00" for i in range(n)]
                values     = [float(v) for v in cleaned_series]
            return timestamps, values

        if self.db is None:
            raise ValueError("Provide either a db instance or cleaned_series.")

        rows = self.db.get_cleaned_data(component, resource, hours=720)
        timestamps = [r["timestamp"] for r in rows]
        values     = [float(r["cleaned_value"]) for r in rows]
        return timestamps, values


# ==============================================================
# QUICK SELF-TEST — run with: python predictor.py
# ==============================================================
if __name__ == "__main__":
    import random

    random.seed(7)
    n = 500  # 500 h of simulated cleaned data

    # Generate realistic sinusoidal load pattern
    ts_list = []
    val_list = []
    base_dt = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(n):
        dt = base_dt + timedelta(hours=i)
        ts_list.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        v = 50 + 20 * math.sin(2 * math.pi * i / 24) + random.gauss(0, 3)
        val_list.append(max(0.0, v))

    pred = CloudPredictor(db=None)
    result = pred.run(
        component="paas_payment",
        resource="acu",
        cleaned_series=[{"timestamp": t, "cleaned_value": v} for t, v in zip(ts_list, val_list)],
    )

    print("\n" + "=" * 55)
    print("predictor.py — self-test results")
    print("=" * 55)
    print(f"  Run ID         : {result['run_id']}")
    print(f"  Best model     : {result['best_model']}")
    print(f"  RMSE table     :")
    for m, r in sorted(result["rmse_table"].items(), key=lambda x: x[1]):
        mark = " ◄ BEST" if m == result["best_model"] else ""
        print(f"      {m:25} → {r:.4f}{mark}")
    print(f"  Forecast rows  : {len(result['forecast'])}")
    print(f"  Feature names  : {result['feature_names']}")
    assert len(result["feature_names"]) == 13, "Expected 13 features"
    assert len(result["forecast"])       == FORECAST_HORIZON
    print("\n  ALL ASSERTIONS PASSED")
    print("  predictor.py is ready.")
