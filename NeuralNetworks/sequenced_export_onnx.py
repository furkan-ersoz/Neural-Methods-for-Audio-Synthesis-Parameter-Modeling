"""
sequenced_export_onnx.py
Export all trained sequential models to ONNX for JUCE InferenceEngine.

Usage:
  python sequenced_export_onnx.py \
    --config synths/SynthV3/Experiments/Exp001/config.yaml

  python sequenced_export_onnx.py \
    --config synths/SynthV3/Experiments/Exp001/config.yaml \
    --frame filter

  python sequenced_export_onnx.py \
    --config synths/SynthV3/Experiments/Exp001/config.yaml \
    --frame door
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_exp_dir"]     = str(config_path.parent)
    cfg["_config_path"] = str(config_path)
    return cfg


def _compute_prev_params_dim(cfg: dict, frame_name: str) -> int:
    from sequenced_dataset import compute_prev_params_dim
    return compute_prev_params_dim(cfg["frames"], frame_name)


def _find_frame_cfg(cfg: dict, frame_name: str) -> dict:
    for f in cfg["frames"]:
        if f["name"] == frame_name:
            return f
    raise ValueError(f"Frame '{frame_name}' not found in config.")


def _output_dir(cfg: dict) -> Path:
    out = Path(cfg["_exp_dir"]) / "exports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _opset(cfg: dict) -> int:
    return int(cfg.get("export", {}).get("opset_version", 17))


# ─────────────────────────────────────────────────────────────────────────────
def _verify_onnx(output_path: Path, dummy_inputs: dict) -> None:
    try:
        import numpy as np
        import onnxruntime as ort

        sess    = ort.InferenceSession(str(output_path))
        np_inputs = {k: v.numpy() for k, v in dummy_inputs.items()}
        outputs = sess.run(None, np_inputs)
        print(f"  ONNX validation OK")
        for i, out in enumerate(outputs):
            print(f"    output[{i}]: shape={out.shape}  "
                  f"min={out.min():.4f}  max={out.max():.4f}")
    except ImportError:
        warnings.warn("onnxruntime not installed — skipping ONNX verification.")


def _save_meta(output_path: Path, meta: dict) -> None:
    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta : {meta_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
def export_door(cfg: dict, checkpoint_path: Path, device: torch.device) -> Path:
    """Export door classifier (model_file from cfg["door"])."""
    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_file = cfg["door"].get("model_file", "model002")
    module     = importlib.import_module(f"models.{model_file}")
    model      = module.build(ckpt["model_cfg"], ckpt["door_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    n_mels   = cfg["features"].get("n_mels", 128)
    door_cfg = ckpt["door_cfg"]

    # door model forward expects (B, n_mels, T) — unsqueezed inside forward
    dummy_mel = torch.randn(1, n_mels, 256, device=device)

    output_path = _output_dir(cfg) / f"{model_file}_door.onnx"
    opset       = _opset(cfg)

    print(f"Exporting door classifier → {output_path.name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy_mel,
            str(output_path),
            opset_version    = opset,
            input_names      = ["mel_full"],
            output_names     = ["bypass_flags", "lfo_count_logits"],
            dynamic_axes     = {
                "mel_full":        {0: "batch", 2: "time"},
                "bypass_flags":    {0: "batch"},
                "lfo_count_logits":{0: "batch"},
            },
            do_constant_folding = True,
        )

    _verify_onnx(output_path, {"mel_full": dummy_mel})

    bypass_cols = list(door_cfg["bypass_columns"])
    meta = {
        "frame_name":        "door",
        "model_file":        model_file,
        "input_names":       ["mel_full"],
        "output_names":      ["bypass_flags", "lfo_count_logits"],
        "n_mels":            n_mels,
        "prev_params_dim":   0,
        "bypass_columns":    bypass_cols,
        "lfo_count_classes": int(door_cfg["lfo_count_classes"]),
        "opset_version":     opset,
    }
    _save_meta(output_path, meta)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
def export_frame(
    cfg:             dict,
    frame_name:      str,
    checkpoint_path: Optional[Path] = None,
    device:          torch.device   = torch.device("cpu"),
) -> Path:
    """Export a single frame predictor (model003–010)."""
    exp_dir   = Path(cfg["_exp_dir"])
    frame_cfg = _find_frame_cfg(cfg, frame_name)
    model_file = frame_cfg["model_file"]

    if checkpoint_path is None:
        checkpoint_path = exp_dir / "checkpoints" / frame_name / "best_full.pt"

    ckpt            = torch.load(checkpoint_path, map_location=device, weights_only=False)
    prev_params_dim = ckpt.get("prev_params_dim", _compute_prev_params_dim(cfg, frame_name))
    saved_frame_cfg = ckpt.get("frame_cfg", frame_cfg)

    module = importlib.import_module(f"models.{model_file}")
    model  = module.build(ckpt["model_cfg"], saved_frame_cfg, prev_params_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    n_mels      = cfg["features"].get("n_mels", 128)
    dummy_full  = torch.randn(1, 1, n_mels, 256, device=device)
    dummy_prev  = torch.randn(1, 1, n_mels, 256, device=device)
    dummy_pp    = torch.zeros(1, max(prev_params_dim, 1), device=device)

    # Determine categorical param names from the model to build output_names
    cont_params = list(saved_frame_cfg.get("params", []))
    cat_params  = {k: list(v) for k, v in saved_frame_cfg.get("categorical_params", {}).items()}
    cat_names   = list(cat_params.keys())
    output_names = ["continuous_out"] + [f"cat_{n}" for n in cat_names]

    # onnx_name from frame_cfg (e.g. "osc_a" → "model003_osc_a.onnx")
    onnx_name   = frame_cfg.get("onnx_filename", f"{model_file}_{frame_name}.onnx")
    output_path = _output_dir(cfg) / onnx_name
    opset       = _opset(cfg)

    print(f"Exporting {frame_name} ({model_file}) → {output_path.name}")

    dynamic_axes: dict = {
        "mel_full":    {0: "batch", 3: "time"},
        "mel_prev":    {0: "batch", 3: "time"},
        "prev_params": {0: "batch"},
        "continuous_out": {0: "batch"},
    }
    for n in cat_names:
        dynamic_axes[f"cat_{n}"] = {0: "batch"}

    # Wrapper to return a flat tuple (required for ONNX multi-output)
    class _FlatWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, mel_full, mel_prev, prev_params):
            cont, cats = self.inner(mel_full, mel_prev, prev_params)
            return (cont,) + tuple(cats[n] for n in cat_names)

    wrapper = _FlatWrapper(model).eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            (dummy_full, dummy_prev, dummy_pp),
            str(output_path),
            opset_version       = opset,
            input_names         = ["mel_full", "mel_prev", "prev_params"],
            output_names        = output_names,
            dynamic_axes        = dynamic_axes,
            do_constant_folding = True,
        )

    verify_inputs = {
        "mel_full": dummy_full,
        "mel_prev": dummy_prev,
    }
    if prev_params_dim > 0:
        verify_inputs["prev_params"] = dummy_pp
    _verify_onnx(output_path, verify_inputs)

    meta = {
        "frame_name":        frame_name,
        "model_file":        model_file,
        "input_names":       ["mel_full", "mel_prev", "prev_params"],
        "output_names":      output_names,
        "n_mels":            n_mels,
        "prev_params_dim":   prev_params_dim,
        "continuous_params": cont_params,
        "categorical_params": cat_params,
        "conditioning":      cfg.get("conditioning", {}),
        "opset_version":     opset,
    }
    _save_meta(output_path, meta)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
def run_export(
    cfg:             dict,
    checkpoint_path: Path,
    device:          torch.device,
    frame_name:      str,
) -> Path:
    """
    Callable entry-point for training scripts (auto-export after training).

    frame_name: "door" or a frame name from config.frames[].name
    """
    if frame_name == "door":
        return export_door(cfg, checkpoint_path, device)
    return export_frame(cfg, frame_name, checkpoint_path, device)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--frame", default="all",
        help="Component to export: 'all', 'door', or frame name (e.g. 'filter')",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg         = load_config(config_path)

    nn_root = Path(__file__).parent
    sys.path.insert(0, str(nn_root))

    device  = torch.device(args.device)
    exp_dir = config_path.parent

    def _door_ckpt() -> Path:
        p = exp_dir / "checkpoints" / "door" / "best_full.pt"
        if not p.exists():
            raise FileNotFoundError(f"Door checkpoint not found: {p}")
        return p

    def _frame_ckpt(frame_name: str) -> Path:
        p = exp_dir / "checkpoints" / frame_name / "best_full.pt"
        if not p.exists():
            raise FileNotFoundError(f"Frame checkpoint not found: {p}")
        return p

    target = args.frame.lower()

    if target == "all":
        # Door
        try:
            export_door(cfg, _door_ckpt(), device)
        except FileNotFoundError as e:
            print(f"  Skipping door: {e}")

        # All frames
        for frame in cfg.get("frames", []):
            fname = frame["name"]
            try:
                export_frame(cfg, fname, _frame_ckpt(fname), device)
            except FileNotFoundError as e:
                print(f"  Skipping {fname}: {e}")

    elif target == "door":
        export_door(cfg, _door_ckpt(), device)

    else:
        export_frame(cfg, target, _frame_ckpt(target), device)


if __name__ == "__main__":
    main()
