"""
dataset.py
Loads a SynthVN dataset (WAV + CSV) and returns PyTorch DataLoaders.
Feature extraction is delegated to feature_extraction.py.
Config is read from an Exp### config.yaml.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from feature_extraction import extract_features, FeatureConfig


# ─────────────────────────────────────────────────────────────────────────────
def collate_variable_length(batch):
    """
    Variable-length spectrogram'lari pad ederek batch olusturur.
    En uzun T'ye gore sifir padding uygulanir (sag taraf).
    Shape: (B, C, T_max)
    """
    features, params = zip(*batch)
    max_t   = max(f.shape[-1] for f in features)
    padded  = torch.zeros(len(features), features[0].shape[0], max_t)
    for i, f in enumerate(features):
        padded[i, :, :f.shape[-1]] = f
    return padded, torch.stack(params)


# ─────────────────────────────────────────────────────────────────────────────
class SynthDataset(Dataset):
    """
    Reads dataset.csv and matching WAV files from a S##D## folder.
    Returns (features, params) pairs as float32 tensors.
    """

    def __init__(
            self,
            dataset_dir: Path,
            param_names: List[str],
            feature_cfg: FeatureConfig,
            cache_dir: Optional[Path] = None,
    ) -> None:
        self.dataset_dir  = Path(dataset_dir)
        self.samples_dir  = self.dataset_dir / "samples"
        self.param_names  = param_names
        self.feature_cfg  = feature_cfg
        self.cache_dir    = cache_dir

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.rows: List[Dict[str, float | str]] = []
        self._load_csv()

    # ── CSV ──────────────────────────────────────────────────────────────────
    def _load_csv(self) -> None:
        csv_path = self.dataset_dir / "dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"Empty dataset: {csv_path}")

    # ── Cache key ─────────────────────────────────────────────────────────────
    def _cache_path(self, filename: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        # Key: filename + feature config hash — config değişince cache geçersiz
        cfg_hash = hashlib.md5(repr(self.feature_cfg).encode()).hexdigest()[:8]
        stem     = Path(filename).stem
        return self.cache_dir / f"{stem}_{cfg_hash}.npy"

    # ── Feature extraction ────────────────────────────────────────────────────
    def _get_features(self, filename: str) -> np.ndarray:
        cache_path = self._cache_path(filename)

        if cache_path and cache_path.exists():
            return np.load(cache_path)

        wav_path = self.samples_dir / filename
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        features = extract_features(wav_path, self.feature_cfg)  # (n_features, T)

        if cache_path:
            np.save(cache_path, features)

        return features

    # ── Dataset protocol ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row      = self.rows[idx]
        filename = row["filename"]

        features = self._get_features(filename)                  # (C, T)
        params   = np.array(
            [float(row[p]) for p in self.param_names], dtype=np.float32
        )                                                        # (num_params,)

        return torch.from_numpy(features).float(), torch.from_numpy(params).float()


# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train / val / test DataLoaders from a parsed config dict.
    Called by train.py and evaluate.py.
    """
    exp_dir     = Path(cfg["_exp_dir"])           # injected by train.py
    synth_root  = (exp_dir / cfg["experiment"]["synth_root"]).resolve()
    dataset_dir = (synth_root / cfg["dataset"]["path"]).resolve()

    param_names = cfg["dataset"]["params"]
    feat_cfg    = FeatureConfig.from_config(cfg["features"])

    cache_dir: Optional[Path] = None
    if cfg["features"].get("cache", False):
        cache_dir = dataset_dir / ".feature_cache"

    full_dataset = SynthDataset(
        dataset_dir=dataset_dir,
        param_names=param_names,
        feature_cfg=feat_cfg,
        cache_dir=cache_dir,
    )

    # ── Train / val / test split ──────────────────────────────────────────────
    n        = len(full_dataset)
    n_train  = int(n * cfg["dataset"]["train_ratio"])
    n_val    = int(n * cfg["dataset"]["val_ratio"])
    n_test   = n - n_train - n_val

    generator = torch.Generator().manual_seed(cfg["dataset"]["seed"])
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    batch_size = cfg["training"]["batch_size"]

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False,
                              collate_fn=collate_variable_length)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False,
                              collate_fn=collate_variable_length)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False,
                              collate_fn=collate_variable_length)

    print(f"Dataset: {n} samples  |  train={n_train}  val={n_val}  test={n_test}")
    return train_loader, val_loader, test_loader