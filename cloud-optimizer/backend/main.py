"""
main.py — FastAPI Bridge
==========================
CloudOptimizer AI  |  Layer 2: Intelligence Engine

Based on: Osypanka & Nawrocki (2022)
"Resource Usage Cost Optimization in Cloud Computing Using Machine Learning"

Endpoints:
    POST /api/predict        — Run ML pipeline on cleaned metrics
    POST /api/optimize       — Run PSO optimizer on predicted demand
    POST /api/full-pipeline  — Detect → Predict → Optimize in one call

Run locally:
    pip install fastapi uvicorn scikit-learn numpy
    uvicorn main:app --reload --port 8000
"""

import os
import sys
import math
import uuid
import random
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from database  import CloudOptimizerDB
from detector  import TwoStageAnomalyFilter
from predictor import CloudPredictor, MIN_TRAIN_ROWS
from optimizer import IntegerPSOOptimizer, TIER_CATALOG, CAPACITY_KEY, _BASELINE_COSTS


# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CloudOptimizer AI",
    description=(
        "Layer 2 Intelligence Engine — Anomaly Detection, ML Forecasting, "
        "and Integer-PSO Cost Optimization for Azure cloud resources.\n\n"
        "Based on Osypanka & Nawrocki (2022)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared DB instance (singleton) ────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cloud_optimizer.db")
db = CloudOptimizerDB(_DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class RawMetricPoint(BaseModel):
    timestamp: str
    value: float


class DetectRequest(BaseModel):
    component:  str = Field(..., example="paas_payment")
    resource:   str = Field(..., example="acu")
    raw_series: Optional[list[RawMetricPoint]] = Field(
        default=None,
        description="Supply raw readings directly; omit to read from DB.",
    )


class PredictRequest(BaseModel):
    component:      str = Field(..., example="paas_payment")
    resource:       str = Field(..., example="acu")
    cleaned_series: Optional[list[RawMetricPoint]] = Field(
        default=None,
        description=(
            "List of cleaned readings. Omit to read from cleaned_metrics table."
        ),
    )


class OptimizeRequest(BaseModel):
    component:     str = Field(..., example="paas_payment")
    demand_series: Optional[list[float]] = Field(
        default=None,
        description=(
            "168-hour demand forecast (one float per hour). "
            "Omit to read latest predictions from DB."
        ),
    )
    n_particles:      int   = Field(default=300,  ge=10,  le=1000)
    n_epochs:         int   = Field(default=500,  ge=10,  le=2000)
    stability_factor: float = Field(default=0.4,  ge=0.0, le=1.0)


class FullPipelineRequest(BaseModel):
    component:    str = Field(..., example="paas_payment")
    resource:     str = Field(..., example="acu")
    raw_series:   Optional[list[RawMetricPoint]] = Field(
        default=None,
        description="Raw readings. Omit to load from DB.",
    )
    n_particles:      int   = Field(default=300,  ge=10, le=1000)
    n_epochs:         int   = Field(default=500,  ge=10, le=2000)
    stability_factor: float = Field(default=0.4,  ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _to_dict_series(points: Optional[list[RawMetricPoint]]) -> Optional[list[dict]]:
    if points is None:
        return None
    return [{"timestamp": p.timestamp, "value": p.value} for p in points]


def _cleaned_to_dict(points: Optional[list[RawMetricPoint]]) -> Optional[list[dict]]:
    if points is None:
        return None
    return [{"timestamp": p.timestamp, "cleaned_value": p.value} for p in points]


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service":  "CloudOptimizer AI",
        "version":  "1.0.0",
        "status":   "running",
        "endpoints": ["/api/predict", "/api/optimize", "/api/full-pipeline"],
    }


@app.get("/api/health", tags=["Health"])
def health():
    try:
        _ = db.get_anomaly_count()
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "database": "ok" if db_ok else "error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: /api/detect (bonus — anomaly detection only)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/detect", tags=["Layer 2: Detect"])
def detect(request: DetectRequest):
    """
    Run the Two-Stage Anomaly Filter on a (component, resource) channel.

    Stage 1 — Exchangeability Martingales (ε=0.9, threshold=20, 24-h z-score)
    Stage 2 — Sliding Median Filter (kernel=5)

    Anomalous values are replaced by the median of the last 6 clean readings
    and are written to `cleaned_metrics` and `anomaly_log` tables.
    """
    raw = _to_dict_series(request.raw_series)

    try:
        det = TwoStageAnomalyFilter(db=db)
        result = det.run(
            component=request.component,
            resource=request.resource,
            raw_series=raw,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status":        "success",
        "component":     request.component,
        "resource":      request.resource,
        "total_points":  result["total"],
        "anomaly_count": result["anomaly_count"],
        "anomaly_indices": result["anomalies"][:50],  # return first 50
        "anomaly_log":     result["log"][:10],         # sample
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: /api/predict
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict", tags=["Layer 2: Predict"])
def predict(request: PredictRequest):
    """
    Train 4 ML models on cleaned metrics, auto-select the best by RMSE,
    and generate a 168-hour forecast.

    Models evaluated:
        BayesianRidge, RandomForest, GradientBoosting, MLPRegressor

    Features used (13 total):
        cyclical time encodings, is_weekend, lag_1/2/24/168,
        rolling stats (mean/std 6h and 24h), raw value.

    Results are saved to the `ml_predictions` table.
    """
    cleaned = _cleaned_to_dict(request.cleaned_series)

    try:
        pred = CloudPredictor(db=db)
        result = pred.run(
            component=request.component,
            resource=request.resource,
            cleaned_series=cleaned,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status":       "success",
        "run_id":       result["run_id"],
        "component":    request.component,
        "resource":     request.resource,
        "best_model":   result["best_model"],
        "rmse_table":   result["rmse_table"],
        "feature_names": result["feature_names"],
        "forecast_hours": len(result["forecast"]),
        "forecast":     result["forecast"][:24],  # first 24 h preview
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: /api/optimize
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/optimize", tags=["Layer 2: Optimize"])
def optimize(request: OptimizeRequest):
    """
    Run Integer-PSO to select the cheapest Azure tier that meets predicted demand.

    Supports components: 'paas_payment' (ACU tiers) and 'saas_database' (DTU tiers).

    PSO settings: n_particles=300, n_epochs=500 (configurable).
    Stability factor F=0.4 prevents frequent tier changes.

    Results are saved to the `optimization_results` and `cost_tracking` tables.
    """
    if request.component not in TIER_CATALOG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown component '{request.component}'. "
                   f"Valid: {list(TIER_CATALOG.keys())}",
        )

    try:
        opt = IntegerPSOOptimizer(
            db=db,
            n_particles=request.n_particles,
            n_epochs=request.n_epochs,
            stability_factor=request.stability_factor,
        )
        result = opt.run(
            component=request.component,
            demand_series=request.demand_series,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status":         "success",
        "opt_id":         result["opt_id"],
        "component":      result["component"],
        "cost_per_hour":  result["cost_per_hour"],
        "monthly_cost":   result["monthly_cost"],
        "baseline_cost":  result["baseline_cost"],
        "savings_pct":    result["savings_pct"],
        "tier_assignments": result["best_tiers"][:24],  # 24-h snapshot
        "pso_details":    result["details"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4: /api/full-pipeline
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/full-pipeline", tags=["Layer 2: Full Pipeline"])
def full_pipeline(request: FullPipelineRequest):
    """
    Run the complete Layer 2 pipeline in a single call:

    1. Anomaly Detection (Two-Stage Filter)  → `cleaned_metrics` table
    2. ML Prediction (best of 4 models)      → `ml_predictions` table
    3. PSO Optimization                      → `optimization_results` table

    Provide raw_series to ingest in-memory data; omit to read from DB.
    """
    pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
    raw = _to_dict_series(request.raw_series)

    # ── Step 1: Anomaly Detection ─────────────────────────────────────────────
    try:
        det = TwoStageAnomalyFilter(db=db)
        det_result = det.run(
            component=request.component,
            resource=request.resource,
            raw_series=raw,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[detect] {e}")

    cleaned_records = det_result.get("cleaned_records", [])

    # ── Step 2: ML Prediction ─────────────────────────────────────────────────
    # Pass the newly cleaned data directly to avoid a DB round-trip
    if len(cleaned_records) < MIN_TRAIN_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Insufficient data for ML training: {len(cleaned_records)} rows "
                f"(need {MIN_TRAIN_ROWS}). "
                "Provide a longer raw_series or accumulate more data in the DB."
            ),
        )

    try:
        pred = CloudPredictor(db=db)
        pred_result = pred.run(
            component=request.component,
            resource=request.resource,
            cleaned_series=cleaned_records,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[predict] {e}")

    # ── Step 3: PSO Optimization ──────────────────────────────────────────────
    try:
        demand_list = [
            f["predicted_value"] for f in pred_result["forecast"]
        ]
        opt = IntegerPSOOptimizer(
            db=db,
            n_particles=request.n_particles,
            n_epochs=request.n_epochs,
            stability_factor=request.stability_factor,
        )
        opt_result = opt.run(
            component=request.component,
            demand_series=demand_list,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[optimize] {e}")

    # ── Assemble response ─────────────────────────────────────────────────────
    return {
        "status":      "success",
        "pipeline_id": pipeline_id,
        "component":   request.component,
        "resource":    request.resource,
        # Step 1
        "detect": {
            "total_points":  det_result["total"],
            "anomaly_count": det_result["anomaly_count"],
            "anomaly_rate_pct": round(
                det_result["anomaly_count"] / max(det_result["total"], 1) * 100, 2
            ),
        },
        # Step 2
        "predict": {
            "run_id":         pred_result["run_id"],
            "best_model":     pred_result["best_model"],
            "rmse_table":     pred_result["rmse_table"],
            "forecast_hours": len(pred_result["forecast"]),
            "forecast_preview": pred_result["forecast"][:24],
        },
        # Step 3
        "optimize": {
            "opt_id":        opt_result["opt_id"],
            "cost_per_hour": opt_result["cost_per_hour"],
            "monthly_cost":  opt_result["monthly_cost"],
            "baseline_cost": opt_result["baseline_cost"],
            "savings_pct":   opt_result["savings_pct"],
            "tier_assignments": opt_result["best_tiers"][:24],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY READ ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/anomalies", tags=["Data"])
def get_anomalies(component: Optional[str] = None):
    """Retrieve anomaly log.  Optionally filter by component."""
    return {"anomalies": db.get_anomalies(component), "count": db.get_anomaly_count(component)}


@app.get("/api/predictions/{component}/{resource}", tags=["Data"])
def get_predictions(component: str, resource: str):
    """Retrieve the best-model predictions for a component/resource channel."""
    return {
        "predictions": db.get_best_predictions(component, resource),
        "component": component,
        "resource": resource,
    }


@app.get("/api/optimization/{component}", tags=["Data"])
def get_optimization(component: str):
    """Retrieve the current active optimization config for a component."""
    result = db.get_active_optimization(component)
    if not result:
        raise HTTPException(status_code=404, detail=f"No active optimization for '{component}'")
    return result


@app.get("/api/savings", tags=["Data"])
def get_savings(component: Optional[str] = None):
    """Retrieve cumulative savings summary."""
    return db.get_total_savings(component)
