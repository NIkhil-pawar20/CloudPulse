import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TMSDataSimulator:
    """
    Simulates realistic cloud resource usage data for the TMS
    (Terminal Management System) — the same IoT payment platform
    tested in the IEEE research paper by Osypanka & Nawrocki (2022).

    Generates hourly readings for 3 cloud components × 6 resources
    with realistic patterns: business-hour peaks, weekend drops,
    random noise, 5% anomaly spike injection, and slow storage growth.

    Used in place of a real Azure Monitor API connection so the
    full ML pipeline can be developed and tested without cloud credentials.
    """

    def __init__(self):
        """Set up base usage levels, component mappings, and random seed."""
        self.bases = {
            'paas_payment_acu':      30,
            'paas_payment_ram':      40,
            'iaas_webpage_acu':      20,
            'iaas_webpage_iops':     35,
            'saas_database_dtu':     45,
            'saas_database_storage': 60,
        }

        self.components = {
            'paas_payment':  ['acu', 'ram'],
            'iaas_webpage':  ['acu', 'iops'],
            'saas_database': ['dtu', 'storage'],
        }

        np.random.seed(42)

    def _calculate_value(self, resource_key, timestamp, hour_index):
        """Calculate one resource value for one specific hour using all 5 patterns."""

        # STEP 1: Get the base value
        base = self.bases[resource_key]

        # STEP 2: Handle storage separately — grows slowly, no spikes
        if resource_key.endswith('_storage'):
            value = base + (hour_index * 0.01) + np.random.normal(0, 1)
            return max(5.0, min(95.0, round(value, 2)))

        # STEP 3: Calculate time_effect (business-hour pattern)
        hour = timestamp.hour
        if 9 <= hour <= 17:
            time_effect = 28
        elif 18 <= hour <= 22:
            time_effect = 8
        elif 6 <= hour <= 8:
            time_effect = 3
        elif hour == 23:
            time_effect = -5
        else:
            time_effect = -15

        # STEP 4: Calculate weekend_effect
        weekday = timestamp.weekday()
        if weekday == 6:
            weekend_effect = -20
        elif weekday == 5:
            weekend_effect = -15
        else:
            weekend_effect = 0

        # STEP 5: Generate noise (small random variation)
        noise = np.random.normal(0, 3)

        # STEP 6: Generate spike (5% chance of anomaly)
        if np.random.random() < 0.05:
            spike = np.random.uniform(40, 80)
        else:
            spike = 0

        # STEP 7: Combine all effects
        value = base + time_effect + weekend_effect + noise + spike

        # STEP 8: Clamp and round
        value = max(5.0, min(100.0, value))
        return round(value, 2)

    def generate_usage(self, hours=720):
        """Generate N hours of simulated data going back from now. Returns a DataFrame with 7 columns."""

        # STEP 1: Calculate timestamps (relative to now)
        now = datetime.now()
        start = now - timedelta(hours=hours)
        timestamps = [start + timedelta(hours=i) for i in range(hours)]

        # STEP 2: Build rows
        rows = []
        for i, ts in enumerate(timestamps):
            row = {'timestamp': ts}
            for component, resources in self.components.items():
                for resource in resources:
                    key = f'{component}_{resource}'
                    row[key] = self._calculate_value(key, ts, i)
            rows.append(row)

        # STEP 3: Return DataFrame
        return pd.DataFrame(rows)

    def get_current_usage(self):
        """Return one single reading representing right now. Values vary on every call."""

        # STEP 1: Use time-based seed so values differ on every call
        np.random.seed(int(datetime.now().microsecond))

        # STEP 2: Generate just 1 hour of data
        df = self.generate_usage(hours=1)

        # STEP 3: Get the single row and convert to dict
        row = df.iloc[-1]
        result = row.to_dict()

        # STEP 4: Convert timestamp to string
        result['timestamp'] = str(result['timestamp'])

        # STEP 5: Return the dict
        return result

    def get_component_data(self, component, hours=720):
        """Return data for one specific component only as a DataFrame."""

        # STEP 1: Generate full data
        df = self.generate_usage(hours)

        # STEP 2: Build column names for this component
        resources = self.components[component]
        cols = [f'{component}_{r}' for r in resources]

        # STEP 3: Return only timestamp + this component's columns
        return df[['timestamp'] + cols]

    def to_db_records(self, df, source='simulated'):
        """Convert wide DataFrame into flat list of dicts matching database.insert_raw_batch() format."""
        records = []
        for _, row in df.iterrows():
            ts = str(row['timestamp'])
            for component, resources in self.components.items():
                for resource in resources:
                    key = f'{component}_{resource}'
                    if key in row.index:
                        records.append({
                            'timestamp': ts,
                            'component': component,
                            'resource':  resource,
                            'value':     float(row[key]),
                            'source':    source,
                        })
        return records


# ==============================================================
# TEST BLOCK — run with: python simulator.py
# ==============================================================
if __name__ == '__main__':
    import os
    import sys
    import time

    print()
    print("=" * 55)
    print("RUNNING SIMULATOR.PY TESTS")
    print("=" * 55)
    print()

    # ----------------------------------------------------------
    # TEST 1 — Basic shape and columns
    # ----------------------------------------------------------
    sim = TMSDataSimulator()
    df = sim.generate_usage(720)

    assert df.shape == (720, 7), f"Expected shape (720, 7), got {df.shape}"
    assert df.columns[0] == 'timestamp', "First column must be 'timestamp'"
    expected_cols = [
        'paas_payment_acu', 'paas_payment_ram',
        'iaas_webpage_acu', 'iaas_webpage_iops',
        'saas_database_dtu', 'saas_database_storage'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    print("[TEST 1] PASS — Shape: (720, 7), all 6 resource columns present")

    # ----------------------------------------------------------
    # TEST 2 — All values within valid range
    # ----------------------------------------------------------
    numeric_cols = expected_cols
    all_in_range = True
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min < 5.0 or col_max > 100.0:
            all_in_range = False
            print(f"  FAIL: {col} out of range — min={col_min}, max={col_max}")

    assert all_in_range, "Some values are outside the 5.0–100.0 range"
    print("[TEST 2] PASS — All values in range 5.0–100.0")
    for col in numeric_cols:
        print(f"  {col + ':':30s} min={df[col].min():5.1f}  max={df[col].max():5.1f}")

    # ----------------------------------------------------------
    # TEST 3 — Business hours pattern
    # ----------------------------------------------------------
    df_temp = df.copy()
    df_temp['hour'] = df_temp['timestamp'].apply(lambda x: x.hour)

    business = df_temp[(df_temp['hour'] >= 10) & (df_temp['hour'] <= 14)]
    night = df_temp[(df_temp['hour'] >= 1) & (df_temp['hour'] <= 4)]

    biz_mean = business['paas_payment_acu'].mean()
    night_mean = night['paas_payment_acu'].mean()

    assert biz_mean > night_mean, (
        f"Business hours mean ({biz_mean:.1f}) should be > night mean ({night_mean:.1f})"
    )
    print("[TEST 3] PASS — Business-hour pattern confirmed")
    print(f"  Business hours (10-14) mean: {biz_mean:.1f}%")
    print(f"  Deep night    ( 1- 4) mean: {night_mean:.1f}%")
    print(f"  Difference: +{biz_mean - night_mean:.1f} percentage points")

    # ----------------------------------------------------------
    # TEST 4 — Weekend pattern
    # ----------------------------------------------------------
    df_temp['weekday'] = df_temp['timestamp'].apply(lambda x: x.weekday())

    weekday_data = df_temp[df_temp['weekday'] < 5]
    weekend_data = df_temp[df_temp['weekday'] >= 5]

    weekday_mean = weekday_data['iaas_webpage_acu'].mean()
    weekend_mean = weekend_data['iaas_webpage_acu'].mean()

    assert weekday_mean > weekend_mean, (
        f"Weekday mean ({weekday_mean:.1f}) should be > weekend mean ({weekend_mean:.1f})"
    )
    print("[TEST 4] PASS — Weekend drop pattern confirmed")
    print(f"  Weekday mean: {weekday_mean:.1f}%")
    print(f"  Weekend mean: {weekend_mean:.1f}%")
    print(f"  Weekend drop: -{weekday_mean - weekend_mean:.1f} percentage points")

    # ----------------------------------------------------------
    # TEST 5 — Anomaly spike injection rate
    # ----------------------------------------------------------
    spike_count = (df['saas_database_dtu'] > 85).sum()
    spike_rate = (spike_count / 720) * 100

    assert 1.0 <= spike_rate <= 15.0, (
        f"Spike rate {spike_rate:.1f}% is outside expected range 1–15%"
    )
    print("[TEST 5] PASS — Spike injection confirmed")
    print(f"  Spikes detected: {spike_count} (values > 85%)")
    print(f"  Spike rate: {spike_rate:.1f}% (expected ~5%)")

    # ----------------------------------------------------------
    # TEST 6 — Storage growth over time
    # ----------------------------------------------------------
    storage_first10 = df['saas_database_storage'].head(10).mean()
    storage_last10 = df['saas_database_storage'].tail(10).mean()
    storage_max = df['saas_database_storage'].max()

    assert storage_last10 > storage_first10, (
        f"Storage should grow: first10={storage_first10:.1f}, last10={storage_last10:.1f}"
    )
    assert storage_max <= 95.0, f"Storage max {storage_max} exceeds 95.0 cap"

    print("[TEST 6] PASS — Storage growth confirmed")
    print(f"  First 10 hours mean: {storage_first10:.1f}%")
    print(f"  Last 10 hours mean:  {storage_last10:.1f}%")
    print(f"  Growth: +{storage_last10 - storage_first10:.1f} percentage points over 720 hours")
    print(f"  Max storage value: {storage_max:.1f}% (must be <= 95.0%)")

    # ----------------------------------------------------------
    # TEST 7 — get_current_usage() returns valid dict
    # ----------------------------------------------------------
    current1 = sim.get_current_usage()
    time.sleep(0.1)
    current2 = sim.get_current_usage()

    assert isinstance(current1, dict), "get_current_usage() must return a dict"
    assert len(current1) == 7, f"Expected 7 keys, got {len(current1)}"
    assert 'timestamp' in current1, "Missing 'timestamp' key"
    assert isinstance(current1['timestamp'], str), "timestamp must be a string"

    for col in expected_cols:
        assert col in current1, f"Missing key: {col}"
        val = current1[col]
        assert isinstance(val, float), f"{col} must be a float, got {type(val)}"
        assert 5.0 <= val <= 100.0, f"{col} value {val} out of range"

    values_differ = current1['paas_payment_acu'] != current2['paas_payment_acu']

    print("[TEST 7] PASS — get_current_usage() works correctly")
    print(f"  Call 1 paas_payment_acu: {current1['paas_payment_acu']:.1f}%")
    print(f"  Call 2 paas_payment_acu: {current2['paas_payment_acu']:.1f}%")
    print(f"  Values differ: {values_differ}")

    # ----------------------------------------------------------
    # TEST 8 — to_db_records() conversion
    # ----------------------------------------------------------
    # Re-seed for reproducibility before generating test data
    np.random.seed(42)
    sim_test = TMSDataSimulator()
    df_small = sim_test.generate_usage(10)
    records = sim_test.to_db_records(df_small)

    assert len(records) == 60, f"Expected 60 records (10×6), got {len(records)}"
    required_keys = {'timestamp', 'component', 'resource', 'value', 'source'}
    assert set(records[0].keys()) == required_keys, (
        f"Record keys {set(records[0].keys())} != expected {required_keys}"
    )
    assert records[0]['source'] == 'simulated', "source must be 'simulated'"
    assert isinstance(records[0]['value'], float), "value must be a float"

    valid_components = {'paas_payment', 'iaas_webpage', 'saas_database'}
    valid_resources = {'acu', 'ram', 'iops', 'dtu', 'storage'}
    assert records[0]['component'] in valid_components, (
        f"Invalid component: {records[0]['component']}"
    )
    assert records[0]['resource'] in valid_resources, (
        f"Invalid resource: {records[0]['resource']}"
    )

    print("[TEST 8] PASS — to_db_records() conversion correct")
    print(f"  10 hours × 6 resources = {len(records)} records")
    print(f"  Sample record: {records[0]}")

    # ----------------------------------------------------------
    # TEST 9 — Full database integration
    # ----------------------------------------------------------
    sys.path.insert(0, '.')
    from database import CloudOptimizerDB

    test_db = '../data/test_sim_integration.db'
    if os.path.exists(test_db):
        os.remove(test_db)

    db = CloudOptimizerDB(db_path=test_db)

    # Generate 24 hours and insert
    np.random.seed(42)
    sim_db = TMSDataSimulator()
    df_24 = sim_db.generate_usage(24)
    records_24 = sim_db.to_db_records(df_24)
    db.insert_raw_batch(records_24)

    # Read back and verify
    result = db.get_raw_data('saas_database', 'dtu', hours=24)
    assert len(result) == 24, f"Expected 24 rows for saas_database/dtu, got {len(result)}"
    assert result[0]['value'] >= 5.0, "Value below minimum"
    assert result[0]['value'] <= 100.0, "Value above maximum"
    assert 'timestamp' in result[0], "Missing timestamp in result"

    # Verify all 3 components × all resources
    for comp, resources in sim_db.components.items():
        for res in resources:
            rows = db.get_raw_data(comp, res, hours=24)
            assert len(rows) == 24, (
                f"Expected 24 rows for {comp}/{res}, got {len(rows)}"
            )

    # Clean up test database
    os.remove(test_db)

    print("[TEST 9] PASS — Database integration confirmed")
    print(f"  24 hours × 6 resources = {len(records_24)} records inserted")
    print("  All 3 components × all 6 resources read back correctly")
    print("  Test database cleaned up")

    # ----------------------------------------------------------
    # PRINT FIRST 5 ROWS — visual confirmation
    # ----------------------------------------------------------
    print()
    print("First 5 rows of generated data:")
    print(df.head().to_string())

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------
    print()
    print("=" * 55)
    print("ALL 9 TESTS PASSED — simulator.py is complete")
    print("=" * 55)
    print()
    print("WHAT THIS FILE DOES:")
    print("  Generates 720 hrs × 6 resources = 4,320 records/run")
    print("  Patterns: business hours, weekends, noise, 5% spikes")
    print("  Storage:  grows +0.01%/hr, capped at 95%")
    print("  Live:     get_current_usage() varies on every call")
    print("  DB-ready: to_db_records() matches insert_raw_batch()")
    print()
    print("NEXT FILE TO BUILD: detector.py")
    print("  detector.py will find and remove those 5% spikes")
    print("  so the ML model only trains on clean, real patterns")
