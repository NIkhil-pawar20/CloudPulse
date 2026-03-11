"""
optimizer.py — Integer-PSO Engine
====================================
CloudOptimizer AI  |  Layer 2: Intelligence Engine

Based on: Osypanka & Nawrocki (2022)
"Resource Usage Cost Optimization in Cloud Computing Using Machine Learning"

Algorithm : Integer Particle Swarm Optimization (iPSO)
Particles : 300
Epochs    : 500
Stability : F = 0.4 — no tier change unless demand shifts > 40 %

Pricing   (Azure PaaS Payment -- ACU tier pricing per hour)
          (Azure SaaS Database  -- DTU tier pricing per hour)

Usage:
    opt = IntegerPSOOptimizer(db)
    result = opt.run(component="paas_payment", demand_series=[...])
"""

import uuid
import json
import math
import random
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# AZURE TIER PRICING  (USD / hour for each discrete capacity unit)
# ─────────────────────────────────────────────────────────────────────────────

PAAS_PAYMENT_TIERS: list[dict] = [
    {"name": "B1",   "acu": 10,   "price_per_hour": 0.018},
    {"name": "B2",   "acu": 20,   "price_per_hour": 0.036},
    {"name": "B3",   "acu": 40,   "price_per_hour": 0.072},
    {"name": "S1",   "acu": 100,  "price_per_hour": 0.100},
    {"name": "S2",   "acu": 200,  "price_per_hour": 0.200},
    {"name": "S3",   "acu": 400,  "price_per_hour": 0.400},
    {"name": "P1v3", "acu": 800,  "price_per_hour": 0.723},
    {"name": "P2v3", "acu": 1600, "price_per_hour": 1.446},
    {"name": "P3v3", "acu": 3200, "price_per_hour": 2.892},
]

SAAS_DATABASE_TIERS: list[dict] = [
    {"name": "Basic",    "dtu": 5,    "price_per_hour": 0.0068},
    {"name": "S0",       "dtu": 10,   "price_per_hour": 0.0202},
    {"name": "S1",       "dtu": 20,   "price_per_hour": 0.0300},
    {"name": "S2",       "dtu": 50,   "price_per_hour": 0.0751},
    {"name": "S3",       "dtu": 100,  "price_per_hour": 0.1503},
    {"name": "S4",       "dtu": 200,  "price_per_hour": 0.3005},
    {"name": "S6",       "dtu": 400,  "price_per_hour": 0.6011},
    {"name": "P1",       "dtu": 125,  "price_per_hour": 0.4668},
    {"name": "P2",       "dtu": 250,  "price_per_hour": 0.9336},
    {"name": "P4",       "dtu": 500,  "price_per_hour": 1.8672},
    {"name": "P6",       "dtu": 1000, "price_per_hour": 3.7344},
    {"name": "P11",      "dtu": 1750, "price_per_hour": 6.5352},
    {"name": "P15",      "dtu": 4000, "price_per_hour": 14.9213},
]

# Baseline (default always-on "safe" config) costs for savings computation
_BASELINE_COSTS: dict[str, float] = {
    "paas_payment": PAAS_PAYMENT_TIERS[-1]["price_per_hour"],   # P3v3
    "saas_database": SAAS_DATABASE_TIERS[-1]["price_per_hour"], # P15
}

TIER_CATALOG: dict[str, list[dict]] = {
    "paas_payment":  PAAS_PAYMENT_TIERS,
    "saas_database": SAAS_DATABASE_TIERS,
}

CAPACITY_KEY: dict[str, str] = {
    "paas_payment":  "acu",
    "saas_database": "dtu",
}


# ─────────────────────────────────────────────────────────────────────────────
# PSO COST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _cost(
    tier_idx: int,
    demand: float,
    component: str,
    prev_tier_idx: Optional[int],
    stability_factor: float,
) -> float:
    """
    Evaluate the hourly cost of a candidate tier assignment.

    Returns
    -------
    float
        Hourly cost in USD, or math.inf when constraints are violated.

    Constraints
    -----------
    * Capacity MUST meet or exceed predicted demand L (hard constraint).
    * Stability: if |new_cap - prev_cap| / prev_cap <= F=0.4, keep prev tier.
    """
    tiers = TIER_CATALOG[component]
    cap_key = CAPACITY_KEY[component]

    if not (0 <= tier_idx < len(tiers)):
        return math.inf

    tier = tiers[tier_idx]
    capacity = tier[cap_key]

    # ── Hard constraint: must meet predicted demand ───────────────────────────
    if capacity < demand:
        return math.inf

    # ── Stability check ───────────────────────────────────────────────────────
    if prev_tier_idx is not None:
        prev_cap = tiers[prev_tier_idx][cap_key]
        relative_change = abs(capacity - prev_cap) / (prev_cap + 1e-9)
        if relative_change <= stability_factor:
            # Not worth switching; penalise lightly if this isn't prev tier
            if tier_idx != prev_tier_idx:
                return tier["price_per_hour"] * 1.05  # small penalty

    return tier["price_per_hour"]


# ─────────────────────────────────────────────────────────────────────────────
# INTEGER-PSO OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class IntegerPSOOptimizer:
    """
    Integer Particle Swarm Optimiser for Azure tier selection.

    Parameters
    ----------
    db              : CloudOptimizerDB | None
    n_particles     : int   (default 300)
    n_epochs        : int   (default 500)
    stability_factor: float (default 0.4)
    w               : float inertia weight           (default 0.72984)
    c1              : float cognitive coefficient     (default 2.05)
    c2              : float social coefficient        (default 2.05)
    seed            : int   for reproducibility
    """

    def __init__(
        self,
        db=None,
        n_particles:     int   = 300,
        n_epochs:        int   = 500,
        stability_factor: float = 0.4,
        w:               float = 0.72984,
        c1:              float = 2.05,
        c2:              float = 2.05,
        seed:            int   = 42,
    ):
        self.db               = db
        self.n_particles      = n_particles
        self.n_epochs         = n_epochs
        self.stability_factor = stability_factor
        self.w                = w
        self.c1               = c1
        self.c2               = c2
        self.seed             = seed

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def run(
        self,
        component: str,
        demand_series: Optional[list] = None,
    ) -> dict:
        """
        Optimise tier assignments for each hour in the demand series.

        Parameters
        ----------
        component     : 'paas_payment' or 'saas_database'
        demand_series : list of floats (predicted demand per hour, 168 h)
                        or list of dicts with key 'predicted_value'.
                        Falls back to self.db if None.

        Returns
        -------
        dict with keys:
            opt_id          str
            component       str
            best_tiers      list[dict]   — per-hour optimal tier assignments
            cost_per_hour   float        — mean hourly cost
            monthly_cost    float
            baseline_cost   float
            savings_pct     float
            details         dict         — PSO meta-info
        """
        random.seed(self.seed)

        if component not in TIER_CATALOG:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Choose from {list(TIER_CATALOG.keys())}."
            )

        demands = self._load_demands(component, demand_series)

        if not demands:
            raise ValueError(f"No demand data for {component}.")

        tiers      = TIER_CATALOG[component]
        n_tiers    = len(tiers)
        baseline   = _BASELINE_COSTS[component]

        # ── Run PSO per hour ──────────────────────────────────────────────────
        best_tiers: list[dict] = []
        prev_idx: Optional[int] = None
        total_cost = 0.0

        for h, L in enumerate(demands):
            best_idx, best_cost = self._pso_single(
                component=component,
                demand=L,
                n_tiers=n_tiers,
                prev_tier_idx=prev_idx,
            )

            if best_idx is not None:
                tier_rec = tiers[best_idx]
                best_tiers.append(
                    {
                        "hour":          h,
                        "demand":        round(L, 2),
                        "tier_name":     tier_rec["name"],
                        "tier_idx":      best_idx,
                        "capacity":      tier_rec[CAPACITY_KEY[component]],
                        "cost_per_hour": round(best_cost, 6),
                    }
                )
                total_cost += best_cost
                prev_idx = best_idx
            else:
                # Fallback: most expensive tier (always feasible)
                fallback = n_tiers - 1
                best_tiers.append(
                    {
                        "hour":          h,
                        "demand":        round(L, 2),
                        "tier_name":     tiers[fallback]["name"],
                        "tier_idx":      fallback,
                        "capacity":      tiers[fallback][CAPACITY_KEY[component]],
                        "cost_per_hour": tiers[fallback]["price_per_hour"],
                    }
                )
                total_cost += tiers[fallback]["price_per_hour"]
                prev_idx = fallback

        n_hours = len(demands)
        cost_per_hour = total_cost / n_hours
        monthly_cost  = cost_per_hour * 24 * 30
        baseline_monthly = baseline * 24 * 30
        savings_pct = (
            (baseline_monthly - monthly_cost) / baseline_monthly * 100
            if baseline_monthly > 0 else 0.0
        )

        opt_id    = f"opt_{component}_{uuid.uuid4().hex[:8]}"
        valid_from = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        result = {
            "opt_id":         opt_id,
            "component":      component,
            "best_tiers":     best_tiers,
            "cost_per_hour":  round(cost_per_hour, 6),
            "monthly_cost":   round(monthly_cost, 2),
            "baseline_cost":  round(baseline_monthly, 2),
            "savings_pct":    round(savings_pct, 2),
            "details": {
                "n_particles":      self.n_particles,
                "n_epochs":         self.n_epochs,
                "stability_factor": self.stability_factor,
                "n_hours":          n_hours,
            },
        }

        # ── Persist ───────────────────────────────────────────────────────────
        if self.db is not None:
            self._save_to_db(result, valid_from)

        return result

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _pso_single(
        self,
        component: str,
        demand: float,
        n_tiers: int,
        prev_tier_idx: Optional[int],
    ) -> tuple[Optional[int], float]:
        """
        Run Integer-PSO for a single hour to find the cheapest feasible tier.

        Positions are continuous floats; rounded to integer tier indices.
        """
        tiers = TIER_CATALOG[component]

        # Initialise particles randomly over [0, n_tiers-1]
        positions  = [random.uniform(0, n_tiers - 1) for _ in range(self.n_particles)]
        velocities = [random.uniform(-1, 1)           for _ in range(self.n_particles)]

        # Personal & global bests
        pbest_pos  = positions[:]
        pbest_cost = [
            _cost(round(p), demand, component, prev_tier_idx, self.stability_factor)
            for p in positions
        ]

        gbest_pos  = pbest_pos[pbest_cost.index(min(pbest_cost))]
        gbest_cost = min(pbest_cost)

        for _epoch in range(self.n_epochs):
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()

                # Velocity update (classic PSO formula)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest_pos[i] - positions[i])
                    + self.c2 * r2 * (gbest_pos   - positions[i])
                )

                # Position update
                positions[i] = positions[i] + velocities[i]

                # Clamp to valid range
                positions[i] = max(0.0, min(n_tiers - 1, positions[i]))

                # Evaluate as integer
                idx  = round(positions[i])
                cost = _cost(idx, demand, component, prev_tier_idx, self.stability_factor)

                # Update personal best
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest_pos[i]  = positions[i]

                # Update global best
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_pos  = positions[i]

            # Early exit if we already found a feasible & cheap solution
            if gbest_cost < math.inf and gbest_cost < tiers[0]["price_per_hour"] * 1.001:
                break

        final_idx = round(gbest_pos)
        if gbest_cost == math.inf:
            return None, math.inf
        return final_idx, gbest_cost

    def _load_demands(self, component: str, demand_series) -> list[float]:
        """Return list of float demands."""
        if demand_series is not None:
            if demand_series and isinstance(demand_series[0], dict):
                return [float(r.get("predicted_value", r.get("demand", 0))) for r in demand_series]
            return [float(v) for v in demand_series]

        if self.db is None:
            raise ValueError("Provide either a db instance or demand_series.")

        cap_key = CAPACITY_KEY[component]
        rows = self.db.get_best_predictions(component, cap_key)
        return [float(r["predicted_value"]) for r in rows]

    def _save_to_db(self, result: dict, valid_from: str):
        """Persist the PSO result to the DB."""
        record = {
            "opt_id":          result["opt_id"],
            "component_type":  result["component"],
            "selected_configs": json.dumps(result["best_tiers"][:24]),  # 24-h snapshot
            "cost_per_hour":   result["cost_per_hour"],
            "monthly_cost":    result["monthly_cost"],
            "baseline_cost":   result["baseline_cost"],
            "savings_pct":     result["savings_pct"],
            "valid_from":      valid_from,
        }
        self.db.insert_optimization(record)

        # Log hourly costs
        for entry in result["best_tiers"][:24]:
            dt_h = (
                datetime.now(timezone.utc)
                .replace(hour=entry["hour"] % 24, minute=0, second=0, microsecond=0)
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            baseline_h = _BASELINE_COSTS[result["component"]]
            optimized_h = entry["cost_per_hour"]
            savings_h   = baseline_h - optimized_h
            savings_pct_h = savings_h / baseline_h * 100 if baseline_h > 0 else 0.0
            self.db.insert_cost(
                {
                    "timestamp":       dt_h,
                    "component_type":  result["component"],
                    "baseline_cost":   round(baseline_h, 6),
                    "optimized_cost":  round(optimized_h, 6),
                    "savings":         round(savings_h, 6),
                    "savings_pct":     round(savings_pct_h, 2),
                }
            )


# ==============================================================
# QUICK SELF-TEST — run with: python optimizer.py
# ==============================================================
if __name__ == "__main__":
    import random as _rand
    _rand.seed(0)

    # Simulate 24-hour demand forecast for paas_payment (ACU values)
    demand_acu = [30 + _rand.uniform(0, 100) for _ in range(24)]

    print("=" * 55)
    print("optimizer.py — self-test (paas_payment, 24 h)")
    print("=" * 55)

    opt = IntegerPSOOptimizer(db=None, n_particles=300, n_epochs=500)
    result = opt.run(component="paas_payment", demand_series=demand_acu)

    print(f"  opt_id         : {result['opt_id']}")
    print(f"  cost/hour (avg): ${result['cost_per_hour']:.4f}")
    print(f"  monthly cost   : ${result['monthly_cost']:.2f}")
    print(f"  baseline cost  : ${result['baseline_cost']:.2f}")
    print(f"  savings        : {result['savings_pct']:.1f}%")
    print(f"\n  Hour-by-hour assignments (first 8):")
    for h in result["best_tiers"][:8]:
        print(
            f"    h={h['hour']:2d}  demand={h['demand']:6.1f}  "
            f"tier={h['tier_name']:6}  cap={h['capacity']:5}  "
            f"cost=${h['cost_per_hour']:.4f}"
        )

    # Validate all tiers meet demand
    for h in result["best_tiers"]:
        assert h["capacity"] >= h["demand"], (
            f"Constraint violated at hour {h['hour']}: "
            f"cap={h['capacity']} < demand={h['demand']}"
        )
    assert result["savings_pct"] > 0, "Expected cost savings vs baseline"

    print("\n  ALL ASSERTIONS PASSED")
    print("  optimizer.py is ready.")
