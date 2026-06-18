"""
feature_extraction_v2.py
WAV dosyasini feature tensor'a donusturur.
Desteklenen tipler: mel_spectrogram | mfcc | stft
Config'den okunan FeatureConfig nesnesi uzerinden calisir.

v2 degisiklikleri (feature_extraction.py'ye gore):
  1. features.stereo: true ise L/R kanallari ayri ayri mel'e cevrilir,
     cikti shape (2, n_mels, T) olur. false ise eski mono davranis.
  2. power_to_db ref=1.0 (mutlak seviye korunur), top_db=80 clamp.
  3. features.normalize: true ise dB degerleri (db + 80) / 80 ile [0, 1]
     araligina donusturulur. Sessizlik = 0.0, maksimum = 1.0. Seviye
     bilgisi korunur (z-score degil, sabit linear rescale).
     Cache hash'i normalize flag'ini icerir — eski cache gecersiz kalir.

ENTEGRASYON NOTU (sequenced_dataset.py icin):
  Stereo modda extract_features() (2, n_mels, T) dondurur, mono modda
  (n_mels, T) (eski ile ayni). Kanal sayisini FeatureConfig.out_channels
  property'sinden oku (mono=1, stereo=2).
  Normalize=True (varsayilan) kullanildiginda MEL_PAD_VALUE = 0.0 (sessizlik).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
FeatureType = Literal["mel_spectrogram", "mfcc", "stft"]


MEL_DB_MIN = -80.0  # top_db=80 ile eslesen minimum dB seviyesi (sessizlik)


@dataclass
class FeatureConfig:
    type:        FeatureType = "mel_spectrogram"
    sample_rate: int         = 44100
    n_mels:      int         = 128
    n_mfcc:      int         = 40
    n_fft:       int         = 2048
    hop_length:  int         = 512
    stereo:      bool        = False
    cache:       bool        = True
    normalize:   bool        = True  # (db + 80) / 80 → [0, 1]; True ise MEL_PAD_VALUE = 0.0

    @property
    def out_channels(self) -> int:
        return 2 if self.stereo else 1

    @property
    def pad_value(self) -> float:
        """Padding icin sessizlik degeri — normalize moduna gore."""
        return 0.0 if self.normalize else MEL_DB_MIN

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
            stereo      = cfg.get("stereo",      False),
            cache       = cfg.get("cache",       True),
            normalize   = cfg.get("normalize",   True),
        )

    def __repr__(self) -> str:
        # Cache key icin deterministik string — normalize dahil (cache bust)
        return (f"FeatureConfig(type={self.type},sr={self.sample_rate},"
                f"n_mels={self.n_mels},n_mfcc={self.n_mfcc},"
                f"n_fft={self.n_fft},hop={self.hop_length},"
                f"stereo={self.stereo},norm={self.normalize})")


# ─────────────────────────────────────────────────────────────────────────────
def extract_features(wav_path: Path, cfg: FeatureConfig) -> np.ndarray:
    """
    WAV dosyasindan feature matrix uretir.

    Donus degeri shape:
      stereo=False: (C, T)    -- mel_spectrogram: C=n_mels, mfcc: C=n_mfcc, stft: C=1+n_fft/2
      stereo=True:  (2, C, T) -- L/R kanallari ayri hesaplanir

    T degiskenmis — model adaptive pooling ile halleder.
    """
    if cfg.stereo:
        y, sr = librosa.load(str(wav_path), sr=cfg.sample_rate, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y], axis=0)
        feat_l = _extract_single_channel(y[0], sr, cfg)
        feat_r = _extract_single_channel(y[1], sr, cfg)
        feat = np.stack([feat_l, feat_r], axis=0)  # (2, C, T)
    else:
        y, sr = librosa.load(str(wav_path), sr=cfg.sample_rate, mono=True)
        feat = _extract_single_channel(y, sr, cfg)  # (C, T)

    return feat.astype(np.float32)


def _extract_single_channel(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    if cfg.type == "mel_spectrogram":
        return _mel_spectrogram(y, sr, cfg)
    elif cfg.type == "mfcc":
        return _mfcc(y, sr, cfg)
    elif cfg.type == "stft":
        return _stft(y, sr, cfg)
    else:
        raise ValueError(f"Unknown feature type: {cfg.type!r}. "
                         f"Choose from: mel_spectrogram, mfcc, stft")


# ─────────────────────────────────────────────────────────────────────────────
def _mel_spectrogram(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    db = librosa.power_to_db(S, ref=1.0, top_db=80)  # (n_mels, T), aralik [-80, 0]
    if cfg.normalize:
        db = (db - MEL_DB_MIN) / (-MEL_DB_MIN)       # [-80,0] → [0,1]; sessizlik=0.0
    return db


def _mfcc(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )                                            # (n_mfcc, T)


def _stft(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    D = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    db = librosa.amplitude_to_db(np.abs(D), ref=1.0, top_db=80)  # (1+n_fft/2, T)
    if cfg.normalize:
        db = (db - MEL_DB_MIN) / (-MEL_DB_MIN)
    return db


# ─────────────────────────────────────────────────────────────────────────────
def save_spectrogram_png(feat: np.ndarray, png_path: Path, cfg: FeatureConfig,
                         title: str = "") -> None:
    """
    Mel spektrogram matrisini PNG olarak kaydeder.
    feat: (n_mels, T) veya (2, n_mels, T) — stereo icin iki panel.
    Normalize=True ise aralik [0,1], False ise [-80,0] dB.
    """
    import matplotlib
    matplotlib.use("Agg")  # GUI gerektirmez
    import matplotlib.pyplot as plt

    vmin, vmax = (0.0, 1.0) if cfg.normalize else (-80.0, 0.0)
    cmap = "magma"

    if feat.ndim == 3:  # stereo: (2, n_mels, T)
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        for ch, (ax, label) in enumerate(zip(axes, ["L", "R"])):
            ax.imshow(feat[ch], aspect="auto", origin="lower",
                      vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
            ax.set_ylabel(f"Mel ({label})")
        axes[0].set_title(title or png_path.stem)
        axes[-1].set_xlabel("Frame")
    else:  # mono: (n_mels, T)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.imshow(feat, aspect="auto", origin="lower",
                  vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax.set_ylabel("Mel bin")
        ax.set_xlabel("Frame")
        ax.set_title(title or png_path.stem)

    label_str = "[0,1] norm" if cfg.normalize else "dB [-80,0]"
    fig.text(0.99, 0.01, label_str, ha="right", va="bottom",
             fontsize=7, color="gray")
    plt.tight_layout()
    plt.savefig(png_path, dpi=80)
    plt.close(fig)


def _extract_one(args: tuple) -> None:
    """Worker function — her WAV icin feature cikart, cache ve PNG kaydet."""
    wav_path, cache_path, png_path, cfg_dict, save_png = args
    cfg = FeatureConfig(**cfg_dict)
    cache_path = Path(cache_path)

    feat_loaded = cache_path.exists()
    feat = np.load(cache_path) if feat_loaded else extract_features(Path(wav_path), cfg)

    if not feat_loaded:
        np.save(cache_path, feat)

    if save_png and png_path and not Path(png_path).exists():
        save_spectrogram_png(feat, Path(png_path), cfg, title=Path(wav_path).stem)


def batch_extract(dataset_dir: Path, cfg: FeatureConfig,
                  cache_dir: Path | None = None,
                  save_png: bool = False,
                  png_dir: Path | None = None,
                  num_workers: int = 0) -> None:
    """
    Tum WAV dosyalarini onceden isleme alir ve cache'e kaydeder.

    num_workers > 0 ise ProcessPoolExecutor ile paralel calisir.
    num_workers = 0 veya 1 → seri (debug icin guvenli).
    save_png=True ise her spektrogram PNG olarak da kaydedilir.
    """
    wav_files = sorted(
        f for d in [dataset_dir / "samples", dataset_dir / "samples_full"]
        if d.exists()
        for f in d.glob("*.wav")
        if not f.name.startswith("._")
    )

    if not wav_files:
        print(f"No WAV files found in {dataset_dir}/samples[_full]")
        return

    if cache_dir is None:
        cache_dir = dataset_dir / ".feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if save_png:
        if png_dir is None:
            png_dir = dataset_dir / "spectrograms"
        png_dir.mkdir(parents=True, exist_ok=True)

    import hashlib
    from dataclasses import asdict
    cfg_hash = hashlib.md5(repr(cfg).encode()).hexdigest()[:8]
    cfg_dict = asdict(cfg)

    print(f"Extracting features for {len(wav_files)} files "
          f"[type={cfg.type}, stereo={cfg.stereo}, normalize={cfg.normalize}, "
          f"workers={num_workers or 1}, cache={cache_dir}]")
    if save_png:
        print(f"  PNG output: {png_dir}")

    jobs = []
    for wav_path in wav_files:
        cache_path = cache_dir / f"{wav_path.stem}_{cfg_hash}.npy"
        png_path_s = str(png_dir / f"{wav_path.stem}.png") if save_png else ""
        already_done = cache_path.exists() and (not save_png or Path(png_path_s).exists())
        if not already_done:
            jobs.append((str(wav_path), str(cache_path), png_path_s, cfg_dict, save_png))

    skipped = len(wav_files) - len(jobs)
    if skipped:
        print(f"  {skipped} dosya cache'den atlanacak, {len(jobs)} islenecek")

    if not jobs:
        print("Feature extraction complete (all cached).")
        return

    done = 0
    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_extract_one, j): j for j in jobs}
            for fut in as_completed(futures):
                fut.result()  # exception'lari yukari tas
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"  {done + skipped}/{len(wav_files)}")
    else:
        for j in jobs:
            _extract_one(j)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done + skipped}/{len(wav_files)}")

    print("Feature extraction complete.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Standalone kullanim:
      python feature_extraction_v2.py --config path/to/Exp###/config.yaml
    """
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True, help="Path to Exp### config.yaml")
    parser.add_argument("--save-png",    action="store_true", default=False)
    parser.add_argument("--png-dir",     default=None)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Paralel worker sayisi (0=seri, >1=paralel)")
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

    num_workers = args.num_workers or cfg.get("features", {}).get("num_workers", 0)
    png_dir     = Path(args.png_dir) if args.png_dir else None

    batch_extract(ds_dir, feat_cfg, cache_dir,
                  save_png=args.save_png, png_dir=png_dir, num_workers=num_workers)
