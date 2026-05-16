import torch
import logging
import os
from thermal_tsdb_dataset import ThermalTSDBDataset, ThermalTSDBDataModule
from train_example import AnomalyTransformer
from attention_store import AttentionStore, extract_mean_attention

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("A3")

# 1. Load model
# We assume the model was trained and saved in sprint3_output/best_model.pt
model = AnomalyTransformer(n_channels=3, window_size=60)
model_path = "sprint3_output/best_model.pt"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        log.info(f"Loaded weights from {model_path}")
    except Exception as e:
        log.error(f"Failed to load weights: {e}. Using random weights.")
else:
    log.warning(f"Model file {model_path} not found. Using random weights for demonstration.")

model.eval()

# 2. Connect DB via AttentionStore
store = AttentionStore()  # Reads DSN from .env

# 3. Load dataset
# Use DataModule to get test patients
dm = ThermalTSDBDataModule(batch_size=1)
train_ids, val_ids, test_ids = dm.split()

# We use the test patients to avoid using training data
dataset = ThermalTSDBDataset(dsn=dm.dsn, patient_ids=test_ids)

# 4. Run inference + store attention
log.info(f"Running inference on {min(50, len(dataset))} windows...")
for i in range(min(50, len(dataset))):
    # Use get_sample_with_meta to get signal, timestamps, and IDs
    try:
        sample = dataset.get_sample_with_meta(i)
        x = sample["signal"].unsqueeze(0)  # Add batch dimension: (1, C, T)
        
        with torch.no_grad():
            # Get mean attention weights (averaged over layers and heads)
            # Using the helper from attention_store.py
            weights = extract_mean_attention(model, x)  # Returns np.ndarray (1, T)
            weights = weights[0]  # Take the first (and only) sample in batch -> (T,)

        # Insert into TimescaleDB
        store.insert(
            window_id  = sample["window_id"],
            patient_id = sample["patient_id"],
            timestamps = sample["timestamps"],
            weights    = weights,
            epoch      = 0
        )
        
        if (i + 1) % 10 == 0:
            log.info(f"  Processed {i+1} windows...")
            
    except Exception as e:
        log.error(f"Error processing window {i}: {e}")
        continue

print(f"\nA3 DONE: Attention maps stored in DB")
print(f"Total rows in attention_maps: {store.count()}")