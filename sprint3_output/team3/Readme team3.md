# Sprint 3 — Team 3 | Topic M6 : Synthetic Thermal Time-Series
**Team SG03 | Sub-team 3 — DB Benchmarks, Schema & Robustness**

---

## Files

| File | What it does |
|---|---|
| `schema_postgres.sql` | Adds `attention_maps` table to PostgreSQL + sample dump |
| `schema_timescaledb.sql` | Adds `attention_maps` hypertable to TimescaleDB |
| `db_pipeline_benchmark.py` | Compares TimescaleDB fetch latency vs raw NPY files (ms/batch) |
| `robustness_test.py` | Injects double-σ noise into test set, measures F1 degradation |

---

## Environment (.env)
```
# PostgreSQL
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=your_password
PG_DB=m6_thermal

# TimescaleDB
TSDB_HOST=localhost
TSDB_PORT=5432
TSDB_USER=postgres
TSDB_PASSWORD=your_password
TSDB_DB=m6_thermal_tsdb

# Sprint 2 data path (absolute recommended on Windows)
NPY_DIR=../../Data-Wrangling/data/processed/npy
```

---

## How to Run

```bash
# 1. Apply schema updates (run once)
psql -U postgres -d m6_thermal      -f sprint3_output/team3/schema_postgres.sql
psql -U postgres -d m6_thermal_tsdb -f sprint3_output/team3/schema_timescaledb.sql

# 2. DB Pipeline Benchmark
python sprint3_output/team3/db_pipeline_benchmark.py

# 3. Robustness Test
python sprint3_output/team3/robustness_test.py
```

Output files saved to `sprint3_output/team3/output/`

---

## Results

### DB Pipeline Benchmark — TimescaleDB vs NPY Files

| Batch size | TSDB ms/batch | NPY ms/batch | Speedup |
|:---:|---:|---:|---:|
| 16  | 492.19  | 366.20  | 0.74x |
| 32  | 1252.20 | 41.44   | 0.03x |
| 64  | 3539.40 | 58.34   | 0.02x |
| 128 | 2177.00 | 122.65  | 0.06x |

**Conclusion:** NPY files are significantly faster for sequential batch loading (up to 60× at batch 64). TimescaleDB adds query parsing overhead unsuitable for tight training loops. However, TimescaleDB remains superior for arbitrary range queries, aggregations, and multi-patient filtering — as shown in Sprint 2 benchmarks (31% faster ingestion, 6.3× compression ratio, 119 MB vs 735 MB).

---

### Robustness Test — Double-σ Noise Injection

| Condition | F1 | Precision | Recall |
|---|:---:|:---:|:---:|
| Clean test set     | 0.0210 | 0.0107 | 0.5000 |
| Noisy test set (2σ) | 0.0057 | 0.0028 | 1.0000 |

| Metric | Value |
|---|:---:|
| Anomaly threshold τ | 0.4279 |
| F1 degradation | **72.86%** |
| σ_original (left, right, asymmetry) | 0.999, 0.999, 0.100 |

**Interpretation:** Under double-σ noise, the model recall increases to 100% (more windows flagged as anomalous) but precision collapses, causing a 72.86% F1 degradation. This confirms the model is sensitive to additive Gaussian noise and that the 85th-percentile threshold needs recalibration under noisy conditions.

---

## Repo Structure

```
Data-Loader/
├── sprint3_output/
│   ├── team3/
│   │   ├── schema_postgres.sql
│   │   ├── schema_timescaledb.sql
│   │   ├── db_pipeline_benchmark.py
│   │   ├── robustness_test.py
│   │   ├── README_team3.md
│   │   └── output/
│   │       ├── db_pipeline_benchmark_results.csv
│   │       ├── db_pipeline_benchmark_results.json
│   │       ├── robustness_results.csv
│   │       └── robustness_results.json
│   └── best_model.pt
└── .env
```