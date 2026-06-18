"""
feature_extraction.py
WAV dosyasini feature tensor'a donusturur.
Desteklenen tipler: mel_spectrogram | mfcc | stft
Config'den okunan FeatureConfig nesnesi uzerinden calisir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
FeatureType = Literal["mel_spectrogram", "mfcc", "stft"]


@dataclass
class FeatureConfig:
    type:        FeatureType = "mel_spectrogram"
    sample_rate: int         = 44100
    n_mels:      int         = 128
    n_mfcc:      int         = 40
    n_fft:       int         = 2048
    hop_length:  int         = 512
    cache:       bool        = True

    # ── Config dict'ten uret ─────────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureConfig":
        return cls(
            type        = cfg.get("type",        "mel_spectrogram"),
            sample_rate = cfg.get("sample_rate", 44100),
            n_mels      = cfg.get("n_mels",      128),
            n_mfcc      = cfg.get("n_mfcc",      40),
            n_fft       = cfg.get("n_fft",       2048),
            hop_length  = cfg.get("hop_length",  512),
            cache       = cfg.get("cache",       True),
        )

    def __repr__(self) -> str:
        # Cache key icin deterministik string
        return (f"FeatureConfig(type={self.type},sr={self.sample_rate},"
                f"n_mels={self.n_mels},n_mfcc={self.n_mfcc},"
                f"n_fft={self.n_fft},hop={self.hop_length})")


# ─────────────────────────────────────────────────────────────────────────────
def extract_features(wav_path: Path, cfg: FeatureConfig) -> np.ndarray:
    """
    WAV dosyasindan feature matrix uretir.

    Donus degeri shape:
      mel_spectrogram: (n_mels, T)
      mfcc:            (n_mfcc, T)
      stft:            (1 + n_fft/2, T)

    T degiskenmis — model adaptive pooling ile halleder.
    """
    y, sr = librosa.load(str(wav_path), sr=cfg.sample_rate, mono=True)

    if cfg.type == "mel_spectrogram":
        feat = _mel_spectrogram(y, sr, cfg)
    elif cfg.type == "mfcc":
        feat = _mfcc(y, sr, cfg)
    elif cfg.type == "stft":
        feat = _stft(y, sr, cfg)
    else:
        raise ValueError(f"Unknown feature type: {cfg.type!r}. "
                         f"Choose from: mel_spectrogram, mfcc, stft")

    return feat.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def _mel_spectrogram(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    # Gucten dB'e — insan algilamasina yakin
    return librosa.power_to_db(S, ref=np.max)   # (n_mels, T)


def _mfcc(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )                                            # (n_mfcc, T)


def _stft(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    D = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    return librosa.amplitude_to_db(np.abs(D), ref=np.max)  # (1+n_fft/2, T)


# ─────────────────────────────────────────────────────────────────────────────
def batch_extract(dataset_dir: Path, cfg: FeatureConfig,
                  cache_dir: Path | None = None) -> None:
    """
    Tum WAV dosyalarini onceden isleme alir ve cache'e kaydeder.
    Opsiyonel: dataset yuklemeden once cagrilabilir.
    """
    samples_dir = dataset_dir / "samples"
    wav_files = sorted(
        f for f in samples_dir.glob("*.wav")
        if not f.name.startswith("._")  # macOS metadata
    )

    if not wav_files:
        print(f"No WAV files found in {samples_dir}")
        return

    if cache_dir is None:
        cache_dir = dataset_dir / ".feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import hashlib
    cfg_hash = hashlib.md5(repr(cfg).encode()).hexdigest()[:8]

    print(f"Extracting features for {len(wav_files)} files "
          f"[type={cfg.type}, cache={cache_dir}]")

    for i, wav_path in enumerate(wav_files):
        cache_path = cache_dir / f"{wav_path.stem}_{cfg_hash}.npy"
        if cache_path.exists():
            continue
        feat = extract_features(wav_path, cfg)
        np.save(cache_path, feat)
        if (i + 1) % 50 == 0 or (i + 1) == len(wav_files):
            print(f"  {i + 1}/{len(wav_files)}")

    print("Feature extraction complete.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Standalone kullanim:
      python feature_extraction.py --config path/to/Exp001/config.yaml
    """
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Exp### config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_dir    = config_path.parent
    synth_root = (exp_dir / cfg["experiment"]["synth_root"]).resolve()
    ds_dir     = (synth_root / cfg["dataset"]["path"]).resolve()
    feat_cfg   = FeatureConfig.from_config(cfg["features"])

    cache_dir: Path | None = None
    if cfg["features"].get("cache", False):
        cache_dir = ds_dir / ".feature_cache"

    batch_extract(ds_dir, feat_cfg, cache_dir)