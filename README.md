# Sprint 3 – Database-to-Model Pipeline
**Topic M6 : Synthetic Thermal Time-Series | Team SG03**

## Objectif

Connecter la base **TimescaleDB** (sprint 2) directement au **DataLoader PyTorch**,
afin que les fenêtres de signal soient lues depuis la base pendant l'entraînement,
sans passer par des fichiers intermédiaires.

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `thermal_tsdb_dataset.py` | Module principal — Dataset + DataModule + benchmark |
| `thermal_npy_dataset.py` | Dataset de référence offline (fichiers `.npy` Sprint 2) |
| `train_example.py` | Boucle d'entraînement complète avec AnomalyTransformer |
| `requirements.txt` | Dépendances Python |

---

## Installation

```bash
pip install -r requirements.txt
```

Copier le fichier d'environnement de Sprint 2 :
```bash
cp /chemin/sprint2/dual_db_ingestion/.env .env
# Vérifier TSDB_HOST, TSDB_PORT, TSDB_USER, TSDB_PASSWORD, TSDB_DB
```

---

## Usage rapide

### 1. Test du DataLoader seul

```bash
python thermal_tsdb_dataset.py --batch-size 32 --max-batches 10
```

Cela va :
- Se connecter à TimescaleDB via `.env`
- Charger les métadonnées de `windows_tsdb`
- Itérer sur 10 batches et afficher shape + latences

### 2. Entraînement complet

```bash
python train_example.py \
    --epochs 20 \
    --batch-size 64 \
    --num-workers 4 \
    --d-model 128 \
    --n-layers 3
```

### 3. Usage dans votre propre script

```python
from thermal_tsdb_dataset import ThermalTSDBDataModule

dm = ThermalTSDBDataModule(
    dsn="postgresql://postgres:password@localhost:5432/m6_thermal_tsdb",
    channels=["left_temperature_norm", "right_temperature_norm", "temp_asymmetry"],
    batch_size=32,
    num_workers=4,
)
train_loader, val_loader, test_loader = dm.get_loaders()

for signals, labels in train_loader:
    # signals : (B, C, T) = (32, 3, 60)  float32
    # labels  : (B,)  int64  —  0=normal, 1=anomalie
    predictions = model(signals)
```

---

## Architecture du pipeline

```
TimescaleDB (hypertable thermal_readings)
        │
        │  SQL : SELECT left_temperature_norm, right_temperature_norm,
        │              temp_asymmetry
        │        FROM thermal_readings
        │        WHERE patient_id = %s
        │          AND timestamp BETWEEN %s AND %s
        │        ORDER BY timestamp ASC
        │
        ▼
ThermalTSDBDataset.__getitem__(idx)
        │  → ndarray float32 (C=3, T=60)
        │  → label int64
        ▼
DataLoader (batch_size=32, num_workers=4)
        │  WeightedRandomSampler → compense déséquilibre 85%/15%
        │  ConnectionPool        → 1 pool psycopg2 par worker
        ▼
AnomalyTransformer
        │  Input proj (C → d_model)
        │  Positional Encoding
        │  TransformerEncoder (multi-head self-attention)
        │  Mean Pooling
        │  MLP Classifier → 2 classes
        ▼
Métriques : F1, Precision, Recall, Accuracy
```

---

## Points clés d'implémentation

### ConnectionPool par worker
PyTorch lance `num_workers` processus séparés via `fork()`.
Les connexions psycopg2 ne sont pas fork-safe, donc **chaque worker crée
son propre pool** (`threading.local`) après le fork :

```python
class ConnectionPool:
    _local = threading.local()
    def _get_pool(self):
        if not hasattr(self._local, "pool"):
            self._local.pool = psycopg2.pool.ThreadedConnectionPool(...)
        return self._local.pool
```

### Requête optimisée (index Sprint 2)
La requête `_fetch_window` exploite l'index composite créé en Sprint 2 :
```sql
-- idx_tr_patient_ts : (patient_id, timestamp DESC)
SELECT left_temperature_norm, right_temperature_norm, temp_asymmetry
FROM thermal_readings
WHERE patient_id = $1
  AND timestamp >= $2 AND timestamp <= $3
ORDER BY timestamp ASC;
```
TimescaleDB utilise le chunk pruning (chunks de 6h) pour ne scanner
que les chunks concernés.

### WeightedRandomSampler
Le dataset Sprint 2 est déséquilibré (~85 % normal, ~15 % anomalie).
Le sampler pondéré compense ce déséquilibre au niveau du batch :
```python
class_weights = 1.0 / bincount(labels)     # [1/0.85, 1/0.15]
sample_weights = class_weights[labels]      # poids par fenêtre
sampler = WeightedRandomSampler(sample_weights, ...)
```

### Split par patient (pas par fenêtre)
Pour éviter la fuite de données (data leakage), le split train/val/test
se fait par `patient_id`, pas par fenêtre individuelle :
```
20 patients → 14 train / 3 val / 3 test
```

---

## Métriques mesurées

| Métrique | Description |
|---|---|
| `db_overhead_pct` | % du temps d'époque consacré à la lecture DB |
| `samples_per_s` | Débit global (samples traités par seconde) |
| `data_fetch_s` | Temps total de fetch DB par époque |
| `compute_s` | Temps total GPU/CPU par époque |

Ces métriques sont exportées dans `sprint3_output/training_report.json`
et servent à rédiger la comparaison DB vs NPY (T7).

---

## Variables d'environnement (.env)

```env
TSDB_HOST=localhost
TSDB_PORT=5432
TSDB_USER=postgres
TSDB_PASSWORD=votre_mot_de_passe
TSDB_DB=m6_thermal_tsdb
```
