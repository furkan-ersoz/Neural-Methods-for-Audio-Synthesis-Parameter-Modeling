"""
export_onnx.py
Kullanim:
  python export_onnx.py --config synths/SynthV1/experiments/Exp001/config.yaml
  python export_onnx.py --config ... --checkpoint last
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_exp_dir"]     = str(config_path.parent)
    cfg["_config_path"] = str(config_path)
    return cfg


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device):
    model_class = cfg["model"]["class"].lower()
    module      = importlib.import_module(f"models.{model_class}")
    num_params  = len(cfg["dataset"]["params"])
    model       = module.build(cfg["model"], output_dim=num_params).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
def get_dummy_input(cfg: dict, device: torch.device) -> torch.Tensor:
    """
    Model icin ornek girdi uretir — ONNX tracing icin gerekli.
    Shape feature tipine gore belirlenir.
    """
    feat_type  = cfg["features"]["type"]
    n_mels     = cfg["features"].get("n_mels",  128)
    n_mfcc     = cfg["features"].get("n_mfcc",   40)
    n_fft      = cfg["features"].get("n_fft",  2048)
    time_frames = 256  # ornek uzunluk

    if feat_type == "mel_spectrogram":
        shape = (1, n_mels, time_frames)
    elif feat_type == "mfcc":
        shape = (1, n_mfcc, time_frames)
    elif feat_type == "stft":
        shape = (1, 1 + n_fft // 2, time_frames)
    else:
        raise ValueError(f"Unknown feature type: {feat_type!r}")

    return torch.randn(*shape, device=device)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  help="Path to Exp### config.yaml")
    parser.add_argument("--checkpoint", default="best", help="best | last (default: best)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg         = load_config(config_path)

    nn_root = Path(__file__).parent
    sys.path.insert(0, str(nn_root))

    device = torch.device("cpu")  # ONNX export CPU'da yapilir

    # ── Checkpoint ────────────────────────────────────────────────────────────
    exp_dir   = config_path.parent
    ckpt_dir  = exp_dir / "checkpoints"
    ckpt_name = "best_full.pt" if args.checkpoint == "best" else "last.pt"
    ckpt_path = ckpt_dir / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading: {ckpt_path}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(cfg, ckpt_path, device)

    # ── Output path ───────────────────────────────────────────────────────────
    export_cfg  = cfg.get("export", {})
    synth_root  = (exp_dir / cfg["experiment"]["synth_root"]).resolve()
    output_dir  = (synth_root / export_cfg.get("output_dir", "../exports")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename    = export_cfg.get("filename", f"{cfg['experiment']['name']}_{cfg['model']['class']}")
    output_path = output_dir / f"{filename}.onnx"

    # ── Dummy input ───────────────────────────────────────────────────────────
    dummy = get_dummy_input(cfg, device)
    print(f"Dummy input shape: {tuple(dummy.shape)}")

    # ── Export ────────────────────────────────────────────────────────────────
    opset = export_cfg.get("opset_version", 17)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["spectrogram"],
        output_names=["params"],
        dynamic_axes={
            "spectrogram": {0: "batch", 2: "time"},  # time axis dinamik
            "params":      {0: "batch"},
        },
        do_constant_folding=True,
    )

    print(f"Exported: {output_path}")
    print(f"  Params : {cfg['dataset']['params']}")
    print(f"  Opset  : {opset}")

    # ── Dogrulama ─────────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np

        sess   = ort.InferenceSession(str(output_path))
        result = sess.run(None, {"spectrogram": dummy.numpy()})
        print(f"  ONNX validation OK — output shape: {result[0].shape}")
    except ImportError:
        print("  onnxruntime not installed — skipping validation.")


if __name__ == "__main__":
    main()