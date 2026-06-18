"""
sequenced_dataset.py
SynthV3 sequential frame prediction pipeline icin dataset modulu.
Mevcut dataset.py'ye dokunmadan V3 multi-frame CSV schemasi destekler.

CSV schema (AllFrames / SelectedFrames modu):
  filename ; <param cols> ;
  bypass_osc_b ; bypass_noise ; bypass_filter ; bypass_reverb ; bypass_delay ;
  active_lfo_count ;
  lfo1_target ; lfo2_target ; lfo3_target ; lfo4_target ;
  filename_frame01 ; ... ; filename_frame12
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from feature_extraction_v2 import extract_features, FeatureConfig


# ─────────────────────────────────────────────────────────────────────────────
# Collate functions
# ─────────────────────────────────────────────────────────────────────────────

# Sessizlik padding degeri: normalize=True (varsayilan) → [0,1] araligi, sessizlik=0.0
# normalize=False kullanilirsa FeatureConfig.pad_value ile manuel override et.
MEL_PAD_VALUE = 0.0


class _CollateVariableLength:
    """DoorDataset collate — pickle'lanabilir sinif (multiprocessing uyumlu)."""
    def __init__(self, pad_value: float = MEL_PAD_VALUE):
        self.pad_value = pad_value

    def __call__(self, batch):
        features, targets = zip(*batch)
        max_t  = max(f.shape[-1] for f in features)
        padded = torch.full((len(features), features[0].shape[0], max_t), self.pad_value)
        for i, f in enumerate(features):
            padded[i, :, :f.shape[-1]] = f
        return padded, torch.stack(targets)


class _CollateFrame:
    """FrameDataset collate — pickle'lanabilir sinif (multiprocessing uyumlu)."""
    def __init__(self, pad_value: float = MEL_PAD_VALUE):
        self.pad_value = pad_value

    def __call__(self, batch):
        mels, prev_params, targets = zip(*batch)
        max_t  = max(m.shape[-1] for m in mels)
        padded = torch.full((len(mels), mels[0].shape[0], max_t), self.pad_value)
        for i, m in enumerate(mels):
            padded[i, :, :m.shape[-1]] = m
        return padded, torch.stack(prev_params), torch.stack(targets)


def make_collate_variable_length(pad_value: float = MEL_PAD_VALUE) -> _CollateVariableLength:
    return _CollateVariableLength(pad_value)


def make_collate_frame(pad_value: float = MEL_PAD_VALUE) -> _CollateFrame:
    return _CollateFrame(pad_value)


# Geriye donuk uyum icin varsayilan collate nesneleri
collate_variable_length = _CollateVariableLength(MEL_PAD_VALUE)
collate_frame           = _CollateFrame(MEL_PAD_VALUE)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_hash(feat_cfg: FeatureConfig) -> str:
    return hashlib.md5(repr(feat_cfg).encode()).hexdigest()[:8]


def _load_features(wav_path: Path,
                   feat_cfg: FeatureConfig,
                   cache_dir: Optional[Path]) -> np.ndarray:
    """WAV dosyasindan feature matrix yukler veya cache'den okur."""
    if cache_dir is not None:
        cache_path = cache_dir / f"{wav_path.stem}_{_cfg_hash(feat_cfg)}.npy"
        if cache_path.exists():
            return np.load(cache_path)
    else:
        cache_path = None

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    feat = extract_features(wav_path, feat_cfg)

    if cache_path is not None:
        np.save(cache_path, feat)

    return feat


def _is_frame_active(row: Dict[str, str], frame_cfg: dict) -> bool:
    """Bu satir icin frame aktif mi?"""
    if frame_cfg.get("always_active", False):
        return True
    if "lfo_index" in frame_cfg:
        return int(row["active_lfo_count"]) >= frame_cfg["lfo_index"]
    # "active_when: active_lfo_count >= N" formatini parse et
    active_when = frame_cfg.get("active_when", "")
    if active_when and "active_lfo_count" in active_when:
        import re
        m = re.search(r"active_lfo_count\s*>=\s*(\d+)", active_when)
        if m:
            return int(row["active_lfo_count"]) >= int(m.group(1))
    if "bypass_column" in frame_cfg:
        return row[frame_cfg["bypass_column"]].strip() == "0"
    # Fallback: frame WAV mevcut mu?
    wav_col = frame_cfg.get("wav_column", "")
    return bool(wav_col and row.get(wav_col, "").strip())


def _encode_param(val_str: str, param_name: str,
                  cat_map: Dict[str, List[str]]) -> float:
    """
    Tek bir CSV param degerini float'a donusturur.
    Kategorik paramlar icin class index dondurur.
    - String deger (lfo_target) -> dogrudan liste aramasiylaindex
    - Float deger + kategorik -> round(v * (N-1)) ile quantize
    """
    if param_name not in cat_map:
        return float(val_str)

    classes = cat_map[param_name]
    try:
        v = float(val_str)
        # Float olarak geldi → quantize to nearest class
        return float(round(v * (len(classes) - 1)))
    except ValueError:
        # String olarak geldi (ornegin lfo_target)
        key = val_str.strip()
        return float(classes.index(key))


def build_global_cat_map(frames: List[dict]) -> Dict[str, List[str]]:
    """Tum frame'lerin categorical_params'ini birlestirerek global bir cat_map uretir."""
    cat_map: Dict[str, List[str]] = {}
    for f in frames:
        cat_map.update(f.get("categorical_params", {}))
    return cat_map


def _param_dim(param_name: str, cat_map: Dict[str, List[str]]) -> int:
    """Bir param'in prev_params vektorundeki boyutu: kategorik -> one-hot (N), continuous -> 1."""
    return len(cat_map[param_name]) if param_name in cat_map else 1


def compute_prev_params_dim(frames: List[dict], frame_name: str) -> int:
    """Verilen frame'den once gelen tum frame'lerin prev_params boyutunu hesaplar."""
    cat_map = build_global_cat_map(frames)
    total = 0
    for f in frames:
        if f["name"] == frame_name:
            break
        for p in f.get("params", []):
            total += _param_dim(p, cat_map)
    return total


def _encode_prev_param(val_str: str, param_name: str,
                       cat_map: Dict[str, List[str]]) -> np.ndarray:
    """
    Tek bir prev-param degerini prev_params vektorune yazilacak parcaya donusturur.
    - Continuous -> shape (1,) skaler deger
    - Categorical -> shape (N,) one-hot vektor (N = sinif sayisi)
    """
    if param_name not in cat_map:
        return np.array([float(val_str)], dtype=np.float32)

    classes = cat_map[param_name]
    n = len(classes)
    onehot = np.zeros(n, dtype=np.float32)
    try:
        v = float(val_str)
        idx = int(round(v * (n - 1)))
    except ValueError:
        idx = classes.index(val_str.strip())
    idx = max(0, min(idx, n - 1))
    onehot[idx] = 1.0
    return onehot


def _build_prev_params(row: Dict[str, str],
                       preceding_frames: List[dict],
                       cat_map: Dict[str, List[str]]) -> np.ndarray:
    """
    Onceki frame'lerin param degerlerini sabit uzunluklu vektor olarak uretir.
    Bypassed frame'ler icin slot sifirlanir (cikarilmaz).
    Kategorik paramlar one-hot, continuous paramlar skaler olarak kodlanir.
    `cat_map`, tum frame'lerin categorical_params'inin birlesimi (global) olmalidir —
    bkz. build_global_cat_map.
    """
    parts: List[np.ndarray] = []
    for frame_cfg in preceding_frames:
        active = _is_frame_active(row, frame_cfg)
        for p in frame_cfg["params"]:
            if active:
                parts.append(_encode_prev_param(row[p], p, cat_map))
            else:
                parts.append(np.zeros(_param_dim(p, cat_map), dtype=np.float32))

    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def _load_csv(dataset_dir: Path) -> List[Dict[str, str]]:
    csv_path = dataset_dir / "dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV bulunamadi: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        # Yorum satirlarini atla
        lines = [ln for ln in f if not ln.startswith("#")]

    reader = csv.DictReader(lines, delimiter=";")
    for row in reader:
        rows.append(dict(row))

    if not rows:
        raise ValueError(f"Bos dataset: {csv_path}")
    return rows


def _validate_sequenced_csv(rows: List[Dict[str, str]]) -> None:
    """Verilen CSV satirlarinin V3 sequenced schema oldugunu dogrular."""
    if "filename_frame01" not in rows[0]:
        raise ValueError(
            "CSV, V3 sequenced schema degil — 'filename_frame01' kolonu bulunamadi. "
            "FinalOnly modunda olusan eski CSV'yi kullanmayin."
        )


# ─────────────────────────────────────────────────────────────────────────────
# DoorDataset
# ─────────────────────────────────────────────────────────────────────────────

class DoorDataset(Dataset):
    """
    model002 (Door Classifier) icin dataset.

    Input:  mel spectrogram of filename_full  → (n_mels, T)
    Target: [bypass_osc_b, bypass_noise, bypass_filter, bypass_reverb,
             bypass_delay, active_lfo_count]  → float32 tensor, shape (6,)

    active_lfo_count float olarak kodlanir; modelin bunu integer olarak
    degerlendirmesi tasarimciya birakilir.
    """

    BYPASS_COLS = [
        "bypass_osc_b",
        "bypass_noise",
        "bypass_filter",
        "bypass_reverb",
        "bypass_delay",
    ]

    def __init__(
            self,
            dataset_dir: Path,
            feat_cfg: FeatureConfig,
            cache_dir: Optional[Path] = None,
    ) -> None:
        # CSV'deki filename degerleri dataset_dir'e goreli (ornegin "samples_full/s00001.wav")
        self.samples_dir = Path(dataset_dir)
        self.feat_cfg    = feat_cfg
        self.cache_dir   = cache_dir

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.rows = _load_csv(Path(dataset_dir))
        _validate_sequenced_csv(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row      = self.rows[idx]
        filename = row["filename"].strip()

        mel = _load_features(self.samples_dir / filename, self.feat_cfg, self.cache_dir)

        bypass_vals = [float(row[c].strip()) for c in self.BYPASS_COLS]
        lfo_count   = float(row["active_lfo_count"].strip())
        target      = np.array(bypass_vals + [lfo_count], dtype=np.float32)

        return torch.from_numpy(mel).float(), torch.from_numpy(target).float()


# ─────────────────────────────────────────────────────────────────────────────
# FrameDataset
# ─────────────────────────────────────────────────────────────────────────────

class FrameDataset(Dataset):
    """
    model003–model010 icin dataset — tek bir frame predictor'u egitir.

    Input A:  mel of filename_full  stack with  mel of prev frame WAV
              Shape: (n_mels, T)  [first frame — no prev]
                  or (2*n_mels, T) [subsequent frames]
              Prev frame WAV yoksa → sifir tensor (ayni shape).
              Training'de prev mel, dropout_p olasilikla sifirlanir.
    Input B:  prev_params_vector  → float32, shape (D,)
              D = sum(len(f["params"]) for f in frames before this one)
              Bypassed frame slotlari sifir; sabit uzunluk.
    Target:   Bu frame'in param degerleri → float32, shape (num_frame_params,)
              Continuous  → [0,1]
              Categorical → integer class index (float olarak)

    Sadece bu frame'in aktif oldugu satirlar yuklenir.
    """

    def __init__(
            self,
            dataset_dir: Path,
            cfg: dict,
            frame_name: str,
            feat_cfg: FeatureConfig,
            cache_dir: Optional[Path] = None,
            training: bool = True,
    ) -> None:
        # CSV'deki filename degerleri dataset_dir'e goreli (ornegin "samples_full/s00001.wav")
        self.samples_dir = Path(dataset_dir)
        self.feat_cfg    = feat_cfg
        self.cache_dir   = cache_dir
        self.training    = training
        self.dropout_p   = cfg.get("conditioning", {}).get("dropout_p", 0.0)
        # Ana mel kaynagi: "full" = tam mix (row["filename"]); "frame" = bu katmana
        # kadarki kumulatif render (frame'in kendi wav_column'u). Geri donuk uyum: "full".
        self.main_render = cfg.get("conditioning", {}).get("main_render", "full")

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        frames = cfg["frames"]

        # Hedef frame'i bul
        frame_indices = [i for i, f in enumerate(frames) if f["name"] == frame_name]
        if not frame_indices:
            raise ValueError(f"Frame '{frame_name}' config'de bulunamadi.")
        self.frame_idx  = frame_indices[0]
        self.frame_cfg  = frames[self.frame_idx]
        self.cat_map: Dict[str, List[str]] = self.frame_cfg.get("categorical_params", {})
        self.global_cat_map: Dict[str, List[str]] = build_global_cat_map(frames)
        self.prev_frames = frames[:self.frame_idx]

        # Onceki frame WAV kolonu (varsa) — "onceki katmana kadarki ses" (mel_prev)
        self.prev_wav_col: Optional[str] = (
            frames[self.frame_idx - 1]["wav_column"]
            if self.frame_idx > 0 else None
        )

        # Bu frame'in kendi (bu katmana kadarki kumulatif) render kolonu —
        # main_render="frame" modunda ana mel girdisi olarak kullanilir.
        self.own_wav_col: Optional[str] = self.frame_cfg.get("wav_column")

        # full_frame_hint: mel_full = tam mix, mel_prev = bu frame'in render'i (dropout ile perdeli)
        if self.main_render == "full_frame_hint" and self.own_wav_col:
            self.prev_wav_col = self.own_wav_col

        # Onceki frame WAV boyutu (sifir tensor uretmek icin n_mels lazim)
        self.n_mels = feat_cfg.n_mels if feat_cfg.type == "mel_spectrogram" else (
            feat_cfg.n_mfcc if feat_cfg.type == "mfcc" else
            (1 + feat_cfg.n_fft // 2)
        )

        all_rows = _load_csv(Path(dataset_dir))
        _validate_sequenced_csv(all_rows)

        # Sadece bu frame'in aktif oldugu satirlari tut
        self.rows = [r for r in all_rows if _is_frame_active(r, self.frame_cfg)]

        if not self.rows:
            raise ValueError(f"Frame '{frame_name}' icin hicbir aktif satir bulunamadi.")

        # prev_params vektor boyutu (kategorik paramlar one-hot olarak sayilir)
        self.prev_dim = compute_prev_params_dim(frames, frame_name)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]

        # ── Ana mel ───────────────────────────────────────────────────────────
        # main_render="full"  -> tam mix (row["filename"])            [blind/endustriyel]
        # main_render="frame" -> bu katmana kadarki kumulatif render  [residual ogrenme]
        if self.main_render == "frame" and self.own_wav_col:
            main_filename = (row.get(self.own_wav_col, "").strip()
                             or row["filename"].strip())
        else:
            main_filename = row["filename"].strip()
        mel_full = _load_features(
            self.samples_dir / main_filename, self.feat_cfg, self.cache_dir,
        )  # (n_mels, T)

        # ── Prev frame mel ────────────────────────────────────────────────────
        if self.prev_wav_col is not None:
            prev_filename = row.get(self.prev_wav_col, "").strip()
            pad_val = self.feat_cfg.pad_value
            if prev_filename:
                mel_prev = _load_features(
                    self.samples_dir / prev_filename, self.feat_cfg, self.cache_dir,
                )
                # T boyutlarini esitle
                T = mel_full.shape[-1]
                if mel_prev.shape[-1] < T:
                    pad = np.full((self.n_mels, T - mel_prev.shape[-1]), pad_val, dtype=np.float32)
                    mel_prev = np.concatenate([mel_prev, pad], axis=-1)
                else:
                    mel_prev = mel_prev[..., :T]
            else:
                mel_prev = np.full_like(mel_full, pad_val)

            # Dropout: training'de prev mel sifirlanabilir
            if self.training and self.dropout_p > 0.0:
                if np.random.random() < self.dropout_p:
                    mel_prev = np.zeros_like(mel_prev)

            mel = np.concatenate([mel_full, mel_prev], axis=0)  # (2*n_mels, T)
        else:
            mel = mel_full  # (n_mels, T)

        # ── Prev params vector ────────────────────────────────────────────────
        prev_params = _build_prev_params(row, self.prev_frames, self.global_cat_map)

        # ── Target params ─────────────────────────────────────────────────────
        target = np.array(
            [_encode_param(row[p], p, self.cat_map) for p in self.frame_cfg["params"]],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(mel).float(),
            torch.from_numpy(prev_params).float(),
            torch.from_numpy(target).float(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builders
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_paths(cfg: dict) -> Tuple[Path, Path]:
    """cfg'den exp_dir ve dataset_dir yollarini cozumler."""
    exp_dir     = Path(cfg["_exp_dir"])
    synth_root  = (exp_dir / cfg["experiment"]["synth_root"]).resolve()
    dataset_dir = (synth_root / cfg["dataset"]["path"]).resolve()
    return exp_dir, dataset_dir


def _split_dataset(full_dataset: Dataset,
                   cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    n       = len(full_dataset)
    n_train = int(n * cfg["dataset"]["train_ratio"])
    n_val   = int(n * cfg["dataset"]["val_ratio"])
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(cfg["dataset"]["seed"])
    return random_split(full_dataset, [n_train, n_val, n_test], generator=generator)


def build_door_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    DoorDataset icin train / val / test DataLoader'lari uretir.
    cfg["_exp_dir"] train.py tarafindan inject edilmis olmali.
    """
    _, dataset_dir = _resolve_paths(cfg)
    feat_cfg       = FeatureConfig.from_config(cfg["features"])

    cache_dir: Optional[Path] = None
    if cfg["features"].get("cache", False):
        cache_dir = dataset_dir / ".feature_cache"

    full_ds = DoorDataset(dataset_dir, feat_cfg, cache_dir)
    train_ds, val_ds, test_ds = _split_dataset(full_ds, cfg)

    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)
    pin_memory  = num_workers > 0
    collate_fn  = make_collate_variable_length(feat_cfg.pad_value)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_fn)

    n = len(full_ds)
    print(f"DoorDataset: {n} samples  |  train={len(train_ds)}  "
          f"val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader


def build_frame_dataloaders(cfg: dict,
                             frame_name: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    FrameDataset icin train / val / test DataLoader'lari uretir.
    """
    _, dataset_dir = _resolve_paths(cfg)
    feat_cfg       = FeatureConfig.from_config(cfg["features"])

    cache_dir: Optional[Path] = None
    if cfg["features"].get("cache", False):
        cache_dir = dataset_dir / ".feature_cache"

    train_ds_full = FrameDataset(dataset_dir, cfg, frame_name, feat_cfg,
                                 cache_dir, training=True)
    # Val / test dataseti training=False (dropout yok) olarak yarat
    eval_ds_full  = FrameDataset(dataset_dir, cfg, frame_name, feat_cfg,
                                 cache_dir, training=False)

    # Split indekslerini esit seed ile belirle
    n       = len(train_ds_full)
    n_train = int(n * cfg["dataset"]["train_ratio"])
    n_val   = int(n * cfg["dataset"]["val_ratio"])
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(cfg["dataset"]["seed"])
    indices   = torch.randperm(n, generator=generator).tolist()

    from torch.utils.data import Subset
    train_ds = Subset(train_ds_full, indices[:n_train])
    val_ds   = Subset(eval_ds_full,  indices[n_train:n_train + n_val])
    test_ds  = Subset(eval_ds_full,  indices[n_train + n_val:])

    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)
    pin_memory  = num_workers > 0
    collate_fn  = make_collate_frame(feat_cfg.pad_value)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate_fn)

    prev_dim    = train_ds_full.prev_dim
    target_dim  = len(train_ds_full.frame_cfg["params"])
    n_input_ch  = feat_cfg.out_channels * (2 if train_ds_full.prev_wav_col else 1)
    print(f"FrameDataset [{frame_name}]: {n} samples  |  "
          f"train={n_train}  val={n_val}  test={n_test}  |  "
          f"mel_channels={n_input_ch}  prev_dim={prev_dim}  target_dim={target_dim}")
    return train_loader, val_loader, test_loader
