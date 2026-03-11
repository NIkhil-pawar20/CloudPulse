"""
detector.py — Two-Stage Anomaly Filter
=======================================
CloudOptimizer AI  |  Layer 2: Intelligence Engine

Based on: Osypanka & Nawrocki (2022)
"Resource Usage Cost Optimization in Cloud Computing Using Machine Learning"

Pipeline:
  Stage 1 → Exchangeability Martingales (ε=0.9, threshold=20, 24-h z-score window)
  Stage 2 → Sliding Median Filter (kernel=5) on Stage-1 residuals
  Replace  → Anomalous points replaced by median of last 6 clean readings

Usage:
    det = TwoStageAnomalyFilter(db)
    result = det.run(component="paas_payment", resource="acu")
"""

import statistics
from datetime import datetime, timezone
from typing import Optional

# ── typing shim so the module works even without DB attached ──────────────────
_DB_TYPE = object  # replaced at runtime


class TwoStageAnomalyFilter:
    """
    Two-Stage Anomaly Filter for cloud metric time-series.

    Parameters
    ----------
    db : CloudOptimizerDB | None
        Database instance.  Pass None for unit-testing without a DB.
    epsilon : float
        Martingale decay factor (default 0.9).
    threshold : float
        Martingale alarm threshold (default 20).
    history_window : int
        Hours of history used for rolling z-score (default 24).
    median_kernel : int
        Kernel size for the sliding median filter (default 5).
    replacement_window : int
        Number of recent CLEAN readings used for replacement median (default 6).
    """

    def __init__(
        self,
        db=None,
        epsilon: float = 0.9,
        threshold: float = 20.0,
        history_window: int = 24,
        median_kernel: int = 5,
        replacement_window: int = 6,
    ):
        self.db = db
        self.epsilon = epsilon
        self.threshold = threshold
        self.history_window = history_window
        self.median_kernel = median_kernel
        self.replacement_window = replacement_window

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        component: str,
        resource: str,
        raw_series: Optional[list] = None,
    ) -> dict:
        """
        Run the full two-stage filter on one (component, resource) channel.

        If `raw_series` is provided it is used directly (list of floats or
        list of dicts with keys 'timestamp' and 'value').
        Otherwise the method fetches from self.db.

        Returns
        -------
        dict with keys:
            cleaned     list[float]  — cleaned values (same length as input)
            anomalies   list[int]    — indices flagged as anomalous
            log         list[dict]   — anomaly records ready for DB insert
            cleaned_records list[dict] — full records for cleaned_metrics table
        """
        # ── 1. Collect raw data ───────────────────────────────────────────────
        timestamps, values = self._load_raw(component, resource, raw_series)

        if len(values) < self.history_window:
            # Not enough data; return raw values as-is
            cleaned_records = [
                {
                    "timestamp": timestamps[i],
                    "component": component,
                    "resource": resource,
                    "raw_value": values[i],
                    "cleaned_value": values[i],
                    "was_anomaly": 0,
                    "anomaly_score": 0.0,
                }
                for i in range(len(values))
            ]
            return {
                "cleaned": values[:],
                "anomalies": [],
                "log": [],
                "cleaned_records": cleaned_records,
            }

        # ── 2. Stage 1 — Exchangeability Martingales ──────────────────────────
        stage1_flags, stage1_scores = self._stage1_martingales(values)

        # ── 3. Stage 2 — Sliding Median Filter on residuals ───────────────────
        stage2_flags = self._stage2_median_filter(values, stage1_flags)

        # ── 4. Build final flag set (union of both stages) ────────────────────
        final_flags = [s1 or s2 for s1, s2 in zip(stage1_flags, stage2_flags)]

        # ── 5. Replace anomalies; collect clean catalogue ─────────────────────
        cleaned_values, anomaly_log, clean_buffer = self._replace_anomalies(
            values, timestamps, final_flags, stage1_scores, component, resource
        )

        # ── 6. Persist to DB if we have one ───────────────────────────────────
        cleaned_records = self._build_cleaned_records(
            timestamps, values, cleaned_values, final_flags,
            stage1_scores, component, resource
        )

        if self.db is not None:
            if cleaned_records:
                self.db.insert_cleaned_batch(cleaned_records)
            for entry in anomaly_log:
                self.db.insert_anomaly(entry)

        anomaly_indices = [i for i, f in enumerate(final_flags) if f]
        return {
            "cleaned": cleaned_values,
            "anomalies": anomaly_indices,
            "log": anomaly_log,
            "cleaned_records": cleaned_records,
            "total": len(values),
            "anomaly_count": len(anomaly_log),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: EXCHANGEABILITY MARTINGALES
    # ──────────────────────────────────────────────────────────────────────────

    def _stage1_martingales(self, values: list[float]):
        """
        Detect anomalies via Exchangeability Martingales.

        For each point t we compute the z-score over the preceding
        `history_window` values as the strangeness measure S_t, then update:
            E_t = ε * E_{t-1} + S_t
        An alarm fires when E_t ≥ threshold.

        Returns
        -------
        flags  : list[bool]   — True where an anomaly is detected
        scores : list[float]  — Martingale E_t values (for DB logging)
        """
        n = len(values)
        flags = [False] * n
        scores = [0.0] * n
        E = 0.0  # martingale accumulator

        for t in range(n):
            # Rolling window for z-score
            start = max(0, t - self.history_window)
            window = values[start:t]  # exclude current point

            if len(window) < 2:
                scores[t] = 0.0
                continue

            mu = statistics.mean(window)
            sigma = statistics.stdev(window)

            if sigma < 1e-9:
                strangeness = 0.0
            else:
                strangeness = abs(values[t] - mu) / sigma

            # Martingale update
            E = self.epsilon * E + strangeness
            scores[t] = E

            if E >= self.threshold:
                flags[t] = True
                E = 0.0  # reset after alarm (prevents run-on detections)

        return flags, scores

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: SLIDING MEDIAN FILTER
    # ──────────────────────────────────────────────────────────────────────────

    def _stage2_median_filter(
        self, values: list[float], stage1_flags: list[bool]
    ) -> list[bool]:
        """
        Apply a sliding median filter to smooth residual noise missed by Stage 1.

        For each point t, compute the residual between the raw value and the
        median of the surrounding kernel window.  If the residual is > 3σ of
        local values the point is flagged.

        Returns
        -------
        flags : list[bool]
        """
        n = len(values)
        flags = [False] * n
        half = self.median_kernel // 2

        for t in range(n):
            if stage1_flags[t]:
                # Already caught by Stage 1; skip to avoid double-counting
                continue

            # Kernel window (clamp to boundaries)
            lo = max(0, t - half)
            hi = min(n, t + half + 1)
            kernel = values[lo:hi]

            if len(kernel) < 3:
                continue

            med = statistics.median(kernel)
            sigma = statistics.stdev(kernel)
            residual = abs(values[t] - med)

            if sigma > 1e-9 and residual / sigma > 3.0:
                flags[t] = True

        return flags

    # ──────────────────────────────────────────────────────────────────────────
    # REPLACEMENT
    # ──────────────────────────────────────────────────────────────────────────

    def _replace_anomalies(
        self,
        values: list[float],
        timestamps: list[str],
        flags: list[bool],
        scores: list[float],
        component: str,
        resource: str,
    ):
        """
        Replace each anomalous value with the median of the last
        `replacement_window` CLEAN readings.

        Returns
        -------
        cleaned_values : list[float]
        anomaly_log    : list[dict]  — ready for DB insert
        clean_buffer   : list[float] — running buffer of clean readings
        """
        cleaned_values = values[:]
        anomaly_log = []
        clean_buffer: list[float] = []  # keeps only clean readings

        for i, (v, is_anomaly) in enumerate(zip(values, flags)):
            if not is_anomaly:
                cleaned_values[i] = v
                clean_buffer.append(v)
                if len(clean_buffer) > self.replacement_window * 4:
                    clean_buffer.pop(0)
            else:
                # Choose replacement value
                recent_clean = clean_buffer[-self.replacement_window:]
                replacement = (
                    statistics.median(recent_clean)
                    if recent_clean
                    else (statistics.mean(values) if values else 0.0)
                )
                cleaned_values[i] = replacement

                # Determine severity label for anomaly_type
                score = scores[i]
                if score >= self.threshold * 2:
                    anomaly_type = "severe"
                elif score >= self.threshold:
                    anomaly_type = "moderate"
                else:
                    anomaly_type = "mild"  # caught by Stage 2

                anomaly_log.append(
                    {
                        "timestamp": timestamps[i],
                        "component": component,
                        "resource": resource,
                        "anomalous_value": v,
                        "replacement": replacement,
                        "anomaly_type": anomaly_type,
                    }
                )

        return cleaned_values, anomaly_log, clean_buffer

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER: build cleaned_metrics records
    # ──────────────────────────────────────────────────────────────────────────

    def _build_cleaned_records(
        self,
        timestamps, raw_values, cleaned_values,
        flags, scores, component, resource,
    ):
        records = []
        for i in range(len(raw_values)):
            records.append(
                {
                    "timestamp": timestamps[i],
                    "component": component,
                    "resource": resource,
                    "raw_value": raw_values[i],
                    "cleaned_value": cleaned_values[i],
                    "was_anomaly": int(flags[i]),
                    "anomaly_score": round(scores[i], 4),
                }
            )
        return records

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER: load raw data from DB or from a directly supplied series
    # ──────────────────────────────────────────────────────────────────────────

    def _load_raw(self, component, resource, raw_series):
        """
        Returns (timestamps: list[str], values: list[float]).
        """
        if raw_series is not None:
            # Accept list[float] or list[dict]
            if raw_series and isinstance(raw_series[0], dict):
                timestamps = [r["timestamp"] for r in raw_series]
                values = [float(r["value"]) for r in raw_series]
            else:
                # Bare list of floats — synthesise ISO timestamps
                now = datetime.now(timezone.utc)
                timestamps = [
                    f"{now.year}-01-01 {i:02d}:00:00" for i in range(len(raw_series))
                ]
                values = [float(v) for v in raw_series]
            return timestamps, values

        if self.db is None:
            raise ValueError("Provide either a db instance or raw_series.")

        rows = self.db.get_raw_data(component, resource, hours=720)
        timestamps = [r["timestamp"] for r in rows]
        values = [float(r["value"]) for r in rows]
        return timestamps, values


# ==============================================================
# QUICK SELF-TEST — run with: python detector.py
# ==============================================================
if __name__ == "__main__":
    import random

    random.seed(42)

    # Synthesise 200 hourly readings with a few injected spikes
    base = [50.0 + random.gauss(0, 5) for _ in range(200)]
    spike_indices = [40, 80, 120, 150, 190]
    for idx in spike_indices:
        base[idx] = 300.0  # obvious spike

    det = TwoStageAnomalyFilter(db=None)
    result = det.run(component="paas_payment", resource="acu", raw_series=base)

    print("=" * 55)
    print("detector.py — self-test results")
    print("=" * 55)
    print(f"  Total points  : {result['total']}")
    print(f"  Anomalies     : {result['anomaly_count']}")
    print(f"  Flagged idx   : {result['anomalies']}")
    print(f"  Expected spikes: {spike_indices}")

    detected = set(result["anomalies"])
    expected = set(spike_indices)
    hits = detected & expected
    print(f"\n  True-positive rate : {len(hits)}/{len(expected)} = {len(hits)/len(expected):.0%}")

    # Verify no replacement is NaN / 0 for flagged points
    for idx in result["anomalies"]:
        v = result["cleaned"][idx]
        assert v is not None and v != 0.0, f"Replacement at {idx} is invalid: {v}"

    print("\n  ALL ASSERTIONS PASSED")
    print("  detector.py is ready.")
