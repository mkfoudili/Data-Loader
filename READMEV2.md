# Sprint 3 — Topic M6 : Synthetic Thermal Time-Series
**Team SG03 | Sub-team 1 — Model Architecture & Training**

---

## What We Did

We built the anomaly detection system that learns what **normal skin temperature looks like**, then flags patients whose signals look different.

The system reads data directly from **TimescaleDB** (no CSV files), trains a deep learning model on healthy patients only, and saves the best model for other teams to use.

---

## How It All Works Together

```
TimescaleDB
    ↓
thermal_tsdb_dataset.py   ← pulls windows from DB, feeds the model
    ↓
train.py                  ← runs the training loop
    ↓  uses ↓
model.py                  ← the brain (TAAE architecture)
loss.py                   ← grades how wrong the model was
    ↓
sprint3_output/
    ├── best_model.pt     ← saved best model (use this for evaluation)
    ├── figure1.png       ← training curves
    ├── table_IV_ablation.csv
    ↓
evaluate.py               ← runs evaluation on test set
    ↓
    ├── table_I_metrics.csv           ← anomaly detection metrics
    └── table_II_reconstruction.csv   ← reconstruction quality metrics
```

---

## Files

### 🆕 Written this Sprint (Team 1)

| File | What it does |
|---|---|
| `model.py` | The TAAE model — encoder, attention, decoder. Import this to load the model. |
| `loss.py` | 4 loss variants: MSE only, MSE+Pattern, MSE+Trend, CPLoss Full. Used during training. |
| `train.py` | Runs training, saves best checkpoint, generates Figure 1 and Table IV. |
| `evaluate.py` | Evaluates the trained model on test set. Computes anomaly detection metrics and reconstruction quality. Outputs Table I and Table II. |

### ✅ Built in Previous Tasks (A1 / A2 / A3)

| File | What it does | Task |
|---|---|---|
| `thermal_tsdb_dataset.py` | Connects TimescaleDB to PyTorch. Feeds windows to the model during training. | A1 |
| `thermal_npy_dataset.py` | Same but reads from .npy files. Used only for the speed benchmark. | A1 |
| `benchmark_A2.py` | Compares DB vs NPY loading speed. Result: NPY is ~110× faster. | A2 |
| `attention_store.py` | Saves attention weights to DB after inference. | A3 |
| `extract_attention_a3.py` | Runs the model on test patients and stores attention maps in DB. | A3 |
| `train_example.py` | Old training script (replaced by train.py — kept for reference). | — |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (default: CPLoss Full)
python train.py

# Run all 4 ablation variants → Table IV
python train.py --ablation

# Evaluate the trained model → Table I (anomaly metrics) & Table II (reconstruction quality)
python evaluate.py --npy-dir ../Data-Wrangling/etl_output/npy

# Store attention maps in DB (after training)
python extract_attention_a3.py
```

---

## Evaluation Outputs

**Table I — Anomaly Detection Metrics** (`table_I_metrics.csv`)
- F1, Precision, Recall scores for anomaly detection
- Individual patient accuracy
- Window/subject anomaly thresholds
- Per-channel false positive rates on healthy subjects
- Confusion matrix (TP, FP, FN, TN)

**Table II — Reconstruction Quality** (`table_II_reconstruction.csv`)
- MAE, RMSE, Pearson correlation, Cosine similarity
- Computed separately for:
  - Training set (all windows)
  - Validation set (all windows)
  - Test healthy windows only
  - Test anomalous windows only

---

## For Other Teams

**Team 2 (Evaluation):**
- Load the model: `from model import TAAE`
- Load weights: `model.load_state_dict(torch.load("sprint3_output/best_model.pt"))`
- Anomaly threshold: 85th percentile of healthy training losses

**Team 3 (Explainability):**
- Attention maps are stored in the `attention_maps` table in TimescaleDB
- Query high-attention windows per patient using `attention_store.py`

---

## Environment (.env)
```
TSDB_HOST=localhost
TSDB_PORT=5433
TSDB_USER=postgres
TSDB_PASSWORD=postgres
TSDB_DB=m6_thermal_tsdb
```
