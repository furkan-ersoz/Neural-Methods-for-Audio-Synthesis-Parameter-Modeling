"""
evaluate.py
Kullanim:
  python evaluate.py --config synths/SynthV1/experiments/Exp001/config.yaml
  python evaluate.py --config ... --checkpoint best   # best.pt (default)
  python evaluate.py --config ... --checkpoint last   # last.pt
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
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

    # best_full.pt varsa state_dict icinden yukle
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_all(model, loader, device):
    all_preds   = []
    all_targets = []
    for features, params in loader:
        features = features.to(device)
        pred     = model(features).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(params.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, targets: np.ndarray,
                    param_names: list[str]) -> dict:
    """
    MSE, MAE, R² — hem global hem per-parametre.
    """
    metrics = {}

    # Global
    metrics["mse_global"] = float(np.mean((preds - targets) ** 2))
    metrics["mae_global"] = float(np.mean(np.abs(preds - targets)))

    # Per-parametre
    per_param = {}
    for i, name in enumerate(param_names):
        p = preds[:, i]
        t = targets[:, i]
        mse = float(np.mean((p - t) ** 2))
        mae = float(np.mean(np.abs(p - t)))
        # R²
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2     = float(1 - ss_res / (ss_tot + 1e-10))
        per_param[name] = {"mse": mse, "mae": mae, "r2": r2}

    metrics["per_param"] = per_param
    return metrics


def print_metrics(metrics: dict, param_names: list[str]) -> None:
    print(f"\n{'─'*50}")
    print(f"  Global MSE : {metrics['mse_global']:.6f}")
    print(f"  Global MAE : {metrics['mae_global']:.6f}")
    print(f"{'─'*50}")
    print(f"  {'Param':<12}  {'MSE':>10}  {'MAE':>10}  {'R²':>8}")
    print(f"  {'─'*44}")
    for name in param_names:
        p = metrics["per_param"][name]
        print(f"  {name:<12}  {p['mse']:>10.6f}  {p['mae']:>10.6f}  {p['r2']:>8.4f}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
def plot_results(preds: np.ndarray, targets: np.ndarray,
                 param_names: list[str], save_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.scatter(targets[:, i], preds[:, i], alpha=0.4, s=10)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = save_dir / "predictions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")

    # Loss curve (train_log.json varsa)
    log_path = save_dir.parent / "logs" / "train_log.json"
    if log_path.exists():
        with open(log_path) as f:
            log_data = json.load(f)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log_data["train_loss"], label="train")
        ax.plot(log_data["val_loss"],   label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training curve")
        ax.legend()
        curve_path = save_dir / "loss_curve.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"Loss curve saved: {curve_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  help="Path to Exp### config.yaml")
    parser.add_argument("--checkpoint", default="best", help="best | last (default: best)")
    parser.add_argument("--split",      default="test", help="train | val | test (default: test)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg         = load_config(config_path)

    nn_root = Path(__file__).parent
    sys.path.insert(0, str(nn_root))
    from dataset import build_dataloaders

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cuda")  if torch.cuda.is_available()         else
        torch.device("cpu")
    )

    # ── Checkpoint ────────────────────────────────────────────────────────────
    exp_dir     = config_path.parent
    ckpt_dir    = exp_dir / "checkpoints"
    ckpt_name   = "best_full.pt" if args.checkpoint == "best" else "last.pt"
    ckpt_path   = ckpt_dir / ckpt_name
    if not ckpt_path.exists():
        # Fallback: best.pt
        ckpt_path = ckpt_dir / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader    = split_map[args.split]
    print(f"Evaluating on: {args.split} split ({len(loader.dataset)} samples)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(cfg, ckpt_path, device)

    # ── Tahmin ────────────────────────────────────────────────────────────────
    preds, targets  = predict_all(model, loader, device)
    param_names     = cfg["dataset"]["params"]
    metrics         = compute_metrics(preds, targets, param_names)

    print_metrics(metrics, param_names)

    # ── Kaydet ────────────────────────────────────────────────────────────────
    results_dir = exp_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / f"metrics_{args.split}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    if cfg.get("evaluation", {}).get("plot", True):
        plot_results(preds, targets, param_names, results_dir)


if __name__ == "__main__":
    main()