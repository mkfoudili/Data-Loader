# Data-Loader Guide (Sprint 3)
**Topic M6 : Synthetic Thermal Time-Series | Team SG03**

This module handles direct TimescaleDB-to-Model data loading for PyTorch.

## 1. Prerequisites

- **Python 3.8+**
- **TimescaleDB** instance running with the following tables (populated in Sprint 2):
  - `subjects`: Patient metadata.
  - `thermal_readings`: High-frequency sensor data.
  - `windows_tsdb`: Pre-computed window segments and labels.
- **Environment File**: A `.env` file in this directory with the following keys:
  ```env
  TSDB_HOST=localhost
  TSDB_PORT=5433
  TSDB_USER=postgres
  TSDB_PASSWORD=postgres
  TSDB_DB=m6_thermal_tsdb
  ```

## 2. Installation

```bash
pip install -r requirements.txt
```

## 3. Execution

### A. Test Database Connection & DataLoader
Verifies if the system can read windows directly from TimescaleDB.
```bash
python thermal_tsdb_dataset.py
```

### B. Train Anomaly Detection Model
Trains the `AnomalyTransformer` model using data fetched from the DB.
```bash
python train_example.py --epochs 10 --batch-size 32
```

### C. Run Performance Benchmark (Task A2)
Compares DB retrieval speed against local `.npy` files.
```bash
# Provide the path to the NPY directory from Sprint 2
python benchmark_A2.py --npy-dir "../Data-Wrangling/etl_output/npy"
```

### D. Extract & Store Attention Maps (Task A3)
Runs inference and stores (window_id, timestamp, weight) in the `attention_maps` table.
```bash
python extract_attention_a3.py
```

## 4. Key Files

- `thermal_tsdb_dataset.py`: Core logic for TimescaleDB-to-PyTorch integration.
- `train_example.py`: Full training loop implementation.
- `attention_store.py`: Database schema and logic for storing attention weights.
- `extract_attention_a3.py`: Script to generate and save attention maps after inference.
- `benchmark_A2.py`: Performance comparison script.