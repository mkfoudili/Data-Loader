"""
evaluate.py

Sprint 3 evaluation for the TAAE skin-temperature anomaly detector.

Outputs:
  - sprint3_output/table_I_metrics.csv
  - sprint3_output/table_II_reconstruction.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import TAAE
from thermal_npy_dataset import DEFAULT_CHANNELS, ThermalNPYDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

WINDOW_SIZE = 60
SEED = 42


class ThermalNPYDatasetWithMetadata(ThermalNPYDataset):
    """ThermalNPYDataset plus patient/window metadata needed for subject metrics."""

    def _load_all(self):
        all_signals = []
        all_labels = []
        all_meta = []

        for pid in self.patient_ids:
            npy_path = self.npy_dir / f"patient_{pid:02d}_windows.npy"
            meta_path = self.npy_dir / f"patient_{pid:02d}_windows_meta.csv"

            if not npy_path.exists():
                log.warning("Missing file: %s", npy_path)
                continue
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing metadata file: {meta_path}")

            arr = np.load(npy_path)
            meta = pd.read_csv(meta_path)

            if "label" not in meta.columns:
                raise ValueError(f"{meta_path} must contain a 'label' column")
            if len(meta) != len(arr):
                raise ValueError(
                    f"Metadata/window count mismatch for patient {pid}: "
                    f"{len(meta)} rows vs {len(arr)} windows"
                )

            arr = arr[:, self.ch_indices, :]
            meta = meta.copy()
            meta["patient_id"] = pid
            if "window_id" not in meta.columns:
                meta["window_id"] = np.arange(len(meta))

            all_signals.append(arr)
            all_labels.append(meta["label"].values)
            all_meta.append(meta)

        if not all_signals:
            raise RuntimeError(
                f"No .npy files found in {self.npy_dir} for patients {self.patient_ids}."
            )

        self.metadata = pd.concat(all_meta, ignore_index=True)
        self.patient_ids_per_window = self.metadata["patient_id"].to_numpy(np.int64)
        self.window_ids = self.metadata["window_id"].to_numpy(np.int64)

        return (
            np.concatenate(all_signals, axis=0).astype(np.float32),
            np.concatenate(all_labels, axis=0).astype(np.int64),
        )


class ReconstructionAccumulator:
    """Streaming aggregate for MAE, RMSE, Pearson r, and cosine similarity."""

    def __init__(self) -> None:
        self.n_windows = 0
        self.n_values = 0
        self.sum_abs = 0.0
        self.sum_sq_err = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: torch.Tensor, x_hat: torch.Tensor) -> None:
        if x.numel() == 0:
            return

        x = x.detach().double().cpu()
        x_hat = x_hat.detach().double().cpu()
        err = x - x_hat

        self.n_windows += int(x.shape[0])
        self.n_values += int(x.numel())
        self.sum_abs += float(err.abs().sum().item())
        self.sum_sq_err += float((err * err).sum().item())
        self.sum_x += float(x.sum().item())
        self.sum_y += float(x_hat.sum().item())
        self.sum_x2 += float((x * x).sum().item())
        self.sum_y2 += float((x_hat * x_hat).sum().item())
        self.sum_xy += float((x * x_hat).sum().item())

    def as_row(self, split_name: str, healthy_windows: int, anomalous_windows: int) -> dict:
        if self.n_values == 0:
            return {
                "Split": split_name,
                "Windows": 0,
                "Healthy Windows": healthy_windows,
                "Anomalous Windows": anomalous_windows,
                "MAE": np.nan,
                "RMSE": np.nan,
                "Pearson Correlation": np.nan,
                "Cosine Similarity": np.nan,
            }

        n = float(self.n_values)
        mae = self.sum_abs / n
        rmse = float(np.sqrt(self.sum_sq_err / n))

        pearson_num = self.sum_xy - (self.sum_x * self.sum_y / n)
        pearson_den_x = self.sum_x2 - (self.sum_x * self.sum_x / n)
        pearson_den_y = self.sum_y2 - (self.sum_y * self.sum_y / n)
        pearson_den = np.sqrt(max(pearson_den_x, 0.0) * max(pearson_den_y, 0.0))
        pearson = pearson_num / pearson_den if pearson_den > 0 else np.nan

        cosine_den = np.sqrt(self.sum_x2) * np.sqrt(self.sum_y2)
        cosine = self.sum_xy / cosine_den if cosine_den > 0 else np.nan

        return {
            "Split": split_name,
            "Windows": self.n_windows,
            "Healthy Windows": healthy_windows,
            "Anomalous Windows": anomalous_windows,
            "MAE": round(float(mae), 8),
            "RMSE": round(float(rmse), 8),
            "Pearson Correlation": round(float(pearson), 8),
            "Cosine Similarity": round(float(cosine), 8),
        }


def discover_patient_ids(npy_dir: Path) -> list[int]:
    patient_ids: list[int] = []
    pattern = re.compile(r"patient_(\d+)_windows\.npy$")
    for path in sorted(npy_dir.glob("patient_*_windows.npy")):
        match = pattern.match(path.name)
        if match:
            patient_ids.append(int(match.group(1)))
    if not patient_ids:
        raise RuntimeError(f"No patient_XX_windows.npy files found in {npy_dir}")
    return patient_ids


def split_patient_ids(
    patient_ids: Iterable[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> tuple[list[int], list[int], list[int]]:
    ids = np.array(list(patient_ids), dtype=int)
    if len(ids) < 3:
        raise ValueError("At least 3 patients are required for train/val/test evaluation")

    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_train = max(1, int(len(ids) * train_ratio))
    n_val = max(1, int(len(ids) * val_ratio))
    if n_train + n_val >= len(ids):
        n_val = max(1, len(ids) - n_train - 1)
    if n_train + n_val >= len(ids):
        n_train = max(1, len(ids) - n_val - 1)

    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train : n_train + n_val].tolist()
    test_ids = ids[n_train + n_val :].tolist()
    return train_ids, val_ids, test_ids


def build_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def load_model(checkpoint_path: Path, n_channels: int, device: torch.device) -> TAAE:
    model = TAAE(n_channels=n_channels, window_size=WINDOW_SIZE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {
            str(k)[len("module.") :] if str(k).startswith("module.") else str(k): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()
    return model


def reconstruction_score(x: torch.Tensor, x_hat: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "mae":
        return (x - x_hat).abs().mean(dim=(1, 2))
    if metric == "mse":
        return ((x - x_hat) ** 2).mean(dim=(1, 2))
    raise ValueError(f"Unsupported score metric: {metric}")


def channel_reconstruction_score(
    x: torch.Tensor, x_hat: torch.Tensor, metric: str
) -> torch.Tensor:
    if metric == "mae":
        return (x - x_hat).abs().mean(dim=2)
    if metric == "mse":
        return ((x - x_hat) ** 2).mean(dim=2)
    raise ValueError(f"Unsupported score metric: {metric}")


@torch.no_grad()
def collect_scores(
    model: TAAE,
    dataset: ThermalNPYDatasetWithMetadata,
    batch_size: int,
    device: torch.device,
    score_metric: str,
) -> dict:
    loader = build_loader(dataset, batch_size)
    losses = []
    channel_losses = []
    labels = []

    for signals, batch_labels in loader:
        signals = signals.to(device)
        x_hat, _ = model(signals)
        losses.append(reconstruction_score(signals, x_hat, score_metric).cpu().numpy())
        channel_losses.append(
            channel_reconstruction_score(signals, x_hat, score_metric).cpu().numpy()
        )
        labels.append(batch_labels.numpy())

    return {
        "losses": np.concatenate(losses),
        "channel_losses": np.concatenate(channel_losses),
        "labels": np.concatenate(labels).astype(np.int64),
        "patient_ids": dataset.patient_ids_per_window.astype(np.int64),
    }


@torch.no_grad()
def reconstruction_rows(
    model: TAAE,
    datasets: dict[str, tuple[ThermalNPYDatasetWithMetadata, Optional[int]]],
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    rows = []

    for split_name, (dataset, label_filter) in datasets.items():
        acc = ReconstructionAccumulator()
        labels_all = dataset._labels
        healthy_windows = int((labels_all == 0).sum())
        anomalous_windows = int((labels_all == 1).sum())

        for signals, labels in build_loader(dataset, batch_size):
            if label_filter is not None:
                mask = labels == label_filter
                if mask.sum().item() == 0:
                    continue
                signals = signals[mask]

            signals = signals.to(device)
            x_hat, _ = model(signals)
            acc.update(signals, x_hat)

        if label_filter == 0:
            row_healthy = int(acc.n_windows)
            row_anomalous = 0
        elif label_filter == 1:
            row_healthy = 0
            row_anomalous = int(acc.n_windows)
        else:
            row_healthy = healthy_windows
            row_anomalous = anomalous_windows

        rows.append(acc.as_row(split_name, row_healthy, row_anomalous))

    return rows


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "F1 (%)": round(100.0 * f1, 2),
        "Precision (%)": round(100.0 * precision, 2),
        "Recall (%)": round(100.0 * recall, 2),
    }


def subject_summary(
    labels: np.ndarray, preds: np.ndarray, patient_ids: np.ndarray
) -> pd.DataFrame:
    rows = []
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        rows.append(
            {
                "patient_id": int(pid),
                "true_pathological": int(labels[mask].sum() > 0),
                "anomaly_pct": 100.0 * float(preds[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def optimize_subject_threshold(
    labels: np.ndarray,
    preds: np.ndarray,
    patient_ids: np.ndarray,
    fallback: float,
) -> float:
    summary = subject_summary(labels, preds, patient_ids)
    if summary["true_pathological"].nunique() < 2:
        log.warning(
            "Validation split has one subject class only; using fallback subject threshold %.2f%%",
            fallback,
        )
        return float(fallback)

    best_threshold = float(fallback)
    best_tuple = (-1.0, -1.0, 0.0)
    for threshold in np.linspace(0.0, 100.0, 1001):
        y_true = summary["true_pathological"].to_numpy(np.int64)
        y_pred = (summary["anomaly_pct"].to_numpy() > threshold).astype(np.int64)
        metrics = binary_metrics(y_true, y_pred)
        accuracy = float((y_true == y_pred).mean())
        candidate = (metrics["F1 (%)"], accuracy, -threshold)
        if candidate > best_tuple:
            best_tuple = candidate
            best_threshold = float(threshold)

    return best_threshold


def subject_accuracy(
    labels: np.ndarray,
    preds: np.ndarray,
    patient_ids: np.ndarray,
    threshold: float,
) -> float:
    summary = subject_summary(labels, preds, patient_ids)
    y_true = summary["true_pathological"].to_numpy(np.int64)
    y_pred = (summary["anomaly_pct"].to_numpy() > threshold).astype(np.int64)
    return round(100.0 * float((y_true == y_pred).mean()), 2)


def channel_index(channels: list[str], prefix: str) -> Optional[int]:
    for idx, channel in enumerate(channels):
        if channel.startswith(prefix):
            return idx
    return None


def channel_fpr_for_healthy_subjects(
    train_scores: dict,
    test_scores: dict,
    channels: list[str],
    percentile: float,
) -> tuple[float, float, str]:
    labels_train = train_scores["labels"]
    labels_test = test_scores["labels"]
    patient_ids_test = test_scores["patient_ids"]

    left_idx = channel_index(channels, "left")
    right_idx = channel_index(channels, "right")
    if left_idx is None or right_idx is None:
        return np.nan, np.nan, "left/right channels not present"

    healthy_train = labels_train == 0
    left_threshold = np.percentile(
        train_scores["channel_losses"][healthy_train, left_idx], percentile
    )
    right_threshold = np.percentile(
        train_scores["channel_losses"][healthy_train, right_idx], percentile
    )

    healthy_subject_ids = []
    for pid in np.unique(patient_ids_test):
        mask = patient_ids_test == pid
        if labels_test[mask].sum() == 0:
            healthy_subject_ids.append(pid)

    if healthy_subject_ids:
        healthy_mask = np.isin(patient_ids_test, healthy_subject_ids)
        population = "test healthy subjects"
    else:
        healthy_mask = labels_test == 0
        population = "test healthy windows (no fully healthy test subject found)"

    if healthy_mask.sum() == 0:
        return np.nan, np.nan, population

    left_fpr = 100.0 * float(
        (test_scores["channel_losses"][healthy_mask, left_idx] > left_threshold).mean()
    )
    right_fpr = 100.0 * float(
        (test_scores["channel_losses"][healthy_mask, right_idx] > right_threshold).mean()
    )
    return round(left_fpr, 2), round(right_fpr, 2), population


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sprint 3 TAAE anomaly detector")
    parser.add_argument("--npy-dir", default="../Data-Wrangling/etl_output/npy")
    parser.add_argument("--checkpoint", default="sprint3_output/best_model.pt")
    parser.add_argument("--out-dir", default="sprint3_output")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--percentile", type=float, default=85.0)
    parser.add_argument("--subject-threshold-fallback", type=float, default=5.0)
    parser.add_argument("--score-metric", choices=["mse", "mae"], default="mse")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patients", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    checkpoint_path = Path(args.checkpoint)
    channels = list(DEFAULT_CHANNELS)

    patient_ids = args.patients if args.patients else discover_patient_ids(npy_dir)
    train_ids, val_ids, test_ids = split_patient_ids(
        patient_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    log.info("Patient split -> train=%s | val=%s | test=%s", train_ids, val_ids, test_ids)

    train_ds = ThermalNPYDatasetWithMetadata(npy_dir, train_ids, channels=channels)
    val_ds = ThermalNPYDatasetWithMetadata(npy_dir, val_ids, channels=channels)
    test_ds = ThermalNPYDatasetWithMetadata(npy_dir, test_ids, channels=channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    model = load_model(checkpoint_path, n_channels=len(channels), device=device)

    train_scores = collect_scores(
        model, train_ds, args.batch_size, device, args.score_metric
    )
    val_scores = collect_scores(model, val_ds, args.batch_size, device, args.score_metric)
    test_scores = collect_scores(
        model, test_ds, args.batch_size, device, args.score_metric
    )

    healthy_train_losses = train_scores["losses"][train_scores["labels"] == 0]
    if len(healthy_train_losses) == 0:
        raise RuntimeError("Training split has no healthy windows for threshold calibration")

    window_threshold = float(np.percentile(healthy_train_losses, args.percentile))
    log.info(
        "Window anomaly threshold (%s, %.1fth percentile of healthy train): %.8f",
        args.score_metric,
        args.percentile,
        window_threshold,
    )

    val_preds = (val_scores["losses"] > window_threshold).astype(np.int64)
    test_preds = (test_scores["losses"] > window_threshold).astype(np.int64)
    optimal_subject_threshold = optimize_subject_threshold(
        val_scores["labels"],
        val_preds,
        val_scores["patient_ids"],
        fallback=args.subject_threshold_fallback,
    )

    anomaly_metrics = binary_metrics(test_scores["labels"], test_preds)
    anomaly_metrics["Algorithm"] = "TAAE"
    anomaly_metrics["Window Loss Metric"] = args.score_metric.upper()
    anomaly_metrics["Window Loss Threshold"] = round(window_threshold, 8)
    anomaly_metrics["Calibration Percentile"] = args.percentile
    anomaly_metrics["Optimal Subject Threshold (%)"] = round(
        optimal_subject_threshold, 2
    )
    anomaly_metrics["Individual Acc. (%)"] = subject_accuracy(
        test_scores["labels"],
        test_preds,
        test_scores["patient_ids"],
        optimal_subject_threshold,
    )

    left_fpr, right_fpr, fpr_population = channel_fpr_for_healthy_subjects(
        train_scores, test_scores, channels, args.percentile
    )
    anomaly_metrics["Left Channel FPR Healthy (%)"] = left_fpr
    anomaly_metrics["Right Channel FPR Healthy (%)"] = right_fpr
    anomaly_metrics["Channel FPR Population"] = fpr_population

    table_i_columns = [
        "Algorithm",
        "Window Loss Metric",
        "Window Loss Threshold",
        "Calibration Percentile",
        "F1 (%)",
        "Precision (%)",
        "Recall (%)",
        "Individual Acc. (%)",
        "Optimal Subject Threshold (%)",
        "TP",
        "FP",
        "FN",
        "TN",
        "Left Channel FPR Healthy (%)",
        "Right Channel FPR Healthy (%)",
        "Channel FPR Population",
    ]
    table_i_row = {column: anomaly_metrics[column] for column in table_i_columns}
    save_csv([table_i_row], out_dir / "table_I_metrics.csv")

    table_ii_rows = reconstruction_rows(
        model,
        {
            "Training": (train_ds, None),
            "Validation": (val_ds, None),
            "Test-Healthy": (test_ds, 0),
            "Test-Anomalous": (test_ds, 1),
        },
        args.batch_size,
        device,
    )
    save_csv(table_ii_rows, out_dir / "table_II_reconstruction.csv")

    print("\nTable I - TAAE anomaly detection")
    for key in [
        "F1 (%)",
        "Precision (%)",
        "Recall (%)",
        "Individual Acc. (%)",
        "Left Channel FPR Healthy (%)",
        "Right Channel FPR Healthy (%)",
    ]:
        print(f"  {key}: {table_i_row[key]}")

    print("\nTable II - reconstruction quality")
    for row in table_ii_rows:
        print(
            f"  {row['Split']}: MAE={row['MAE']}, RMSE={row['RMSE']}, "
            f"r={row['Pearson Correlation']}, cosine={row['Cosine Similarity']}"
        )


if __name__ == "__main__":
    main()
