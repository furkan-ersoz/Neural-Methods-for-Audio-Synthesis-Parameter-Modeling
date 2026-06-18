"""
sequenced_train_frames.py
Phase 2 — Frame Predictor training (model003–model010).

Usage:
  python sequenced_train_frames.py \
    --config synths/SynthV3/Experiments/Exp001/config.yaml \
    --frame filter

  python sequenced_train_frames.py \
    --config synths/SynthV3/Experiments/Exp001/config.yaml \
    --frame all
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_exp_dir"]     = str(config_path.parent)
    cfg["_config_path"] = str(config_path)
    return cfg


def build_optimizer(cfg: dict, model: nn.Module, lr: float | None = None) -> torch.optim.Optimizer:
    opt_type = cfg["training"].get("optimizer", "adam").lower()
    lr       = lr if lr is not None else cfg["training"].get("learning_rate", 1e-3)
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type!r}")


def build_lr_scheduler(opt: torch.optim.Optimizer, sched_type: str, epochs: int,
                       plateau_patience: int = 10):
    if sched_type == "none":
        return None
    elif sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-6
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {sched_type!r}")


def _resolve_frame_training(cfg: dict, frame_cfg: dict) -> dict:
    """Merge global training config with per-frame train_overrides."""
    train_cfg = cfg["training"]
    overrides = frame_cfg.get("train_overrides", {})
    return {
        "learning_rate": overrides.get("learning_rate", train_cfg.get("learning_rate", 1e-3)),
        "patience":      overrides.get("patience",
                                       train_cfg.get("early_stopping", {}).get("patience", 999)),
        "lr_scheduler":  overrides.get("lr_scheduler", train_cfg.get("lr_scheduler", "cosine")),
        "grad_clip":     overrides.get("grad_clip", train_cfg.get("grad_clip")),
    }


def _compute_prev_params_dim(cfg: dict, frame_name: str) -> int:
    from sequenced_dataset import compute_prev_params_dim
    return compute_prev_params_dim(cfg["frames"], frame_name)


def _find_frame_cfg(cfg: dict, frame_name: str) -> dict:
    for f in cfg["frames"]:
        if f["name"] == frame_name:
            return f
    raise ValueError(f"Frame '{frame_name}' not found in config frames list.")


def _split_mel(mel: torch.Tensor, n_mels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B, C, T) → (B, 1, n_mels, T), (B, 1, n_mels, T)"""
    mel_full = mel[:, :n_mels, :].unsqueeze(1)
    if mel.shape[1] > n_mels:
        mel_prev = mel[:, n_mels:, :].unsqueeze(1)
    else:
        mel_prev = torch.zeros_like(mel_full)
    return mel_full, mel_prev


def _split_target(
    target: torch.Tensor,
    frame_cfg: dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """target (B, P) → cont_target (B, n_cont), cat_targets {name: (B,) long}"""
    params     = frame_cfg["params"]
    cat_params = frame_cfg.get("categorical_params", {})

    cont_idx = [i for i, p in enumerate(params) if p not in cat_params]
    cat_idx  = {p: i for i, p in enumerate(params) if p in cat_params}

    cont_target = target[:, cont_idx] if cont_idx else torch.zeros(target.size(0), 0,
                                                                     device=target.device)
    cat_targets = {p: target[:, i].long() for p, i in cat_idx.items()}
    return cont_target, cat_targets


# ─────────────────────────────────────────────────────────────────────────────
def _train_epoch(
    model,
    loader,
    optimizer,
    compute_loss_fn,
    frame_cfg: dict,
    n_mels: int,
    cat_weight: float,
    device: torch.device,
    grad_clip: float | None = None,
) -> dict:
    model.train()
    total_loss = 0.0
    total_mse  = 0.0
    total_grad_norm = 0.0
    n_batches  = 0
    cat_totals: Dict[str, float] = {}

    for mel, prev_params, target in loader:
        mel         = mel.to(device)
        prev_params = prev_params.to(device)
        target      = target.to(device)

        mel_full, mel_prev = _split_mel(mel, n_mels)
        cont_target, cat_targets = _split_target(target, frame_cfg)
        cat_targets = {k: v.to(device) for k, v in cat_targets.items()}

        optimizer.zero_grad()
        cont_out, cat_outs = model(mel_full, mel_prev, prev_params)
        loss = compute_loss_fn(cont_out, cat_outs, cont_target, cat_targets,
                               cat_weight=cat_weight,
                               per_param_loss_weight=model.per_param_loss_weight)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip if grad_clip else float("inf")
        )
        total_grad_norm += grad_norm.item()
        n_batches += 1
        optimizer.step()

        B = mel.size(0)
        with torch.no_grad():
            total_loss += loss.item() * B
            if cont_out.numel() > 0:
                from models.model011 import continuous_predictions
                cont_values = continuous_predictions(cont_out)
                total_mse += F.mse_loss(cont_values, cont_target).item() * B
            for name, logits in cat_outs.items():
                ce = F.cross_entropy(logits, cat_targets[name].long()).item()
                cat_totals[name] = cat_totals.get(name, 0.0) + ce * B

    n = len(loader.dataset)
    metrics = {"loss": total_loss / n, "mse": total_mse / n,
               "grad_norm": total_grad_norm / max(n_batches, 1)}
    for name, v in cat_totals.items():
        metrics[f"ce_{name}"] = v / n
    return metrics


@torch.no_grad()
def _eval_epoch(
    model,
    loader,
    compute_loss_fn,
    frame_cfg: dict,
    n_mels: int,
    cat_weight: float,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_mse  = 0.0
    cat_totals: Dict[str, float] = {}

    for mel, prev_params, target in loader:
        mel         = mel.to(device)
        prev_params = prev_params.to(device)
        target      = target.to(device)

        mel_full, mel_prev = _split_mel(mel, n_mels)
        cont_target, cat_targets = _split_target(target, frame_cfg)
        cat_targets = {k: v.to(device) for k, v in cat_targets.items()}

        cont_out, cat_outs = model(mel_full, mel_prev, prev_params)
        loss = compute_loss_fn(cont_out, cat_outs, cont_target, cat_targets,
                               cat_weight=cat_weight,
                               per_param_loss_weight=model.per_param_loss_weight)

        B = mel.size(0)
        total_loss += loss.item() * B
        if cont_out.numel() > 0:
            from models.model011 import continuous_predictions
            cont_values = continuous_predictions(cont_out)
            total_mse += F.mse_loss(cont_values, cont_target).item() * B
        for name, logits in cat_outs.items():
            ce = F.cross_entropy(logits, cat_targets[name].long()).item()
            cat_totals[name] = cat_totals.get(name, 0.0) + ce * B

    n = len(loader.dataset)
    if n == 0:
        return {"loss": 0.0, "mse": 0.0, "mae": 0.0}
    metrics = {"loss": total_loss / n, "mse": total_mse / n}
    for name, v in cat_totals.items():
        metrics[f"ce_{name}"] = v / n
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def _train_frame(cfg: dict, frame_name: str, device: torch.device,
                 use_lr_scheduler: bool = False) -> None:
    from sequenced_dataset import build_frame_dataloaders

    exp_dir  = Path(cfg["_exp_dir"])
    ckpt_dir = exp_dir / "checkpoints" / frame_name
    log_dir  = exp_dir / "logs" / frame_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    frame_cfg       = _find_frame_cfg(cfg, frame_name)
    prev_params_dim = _compute_prev_params_dim(cfg, frame_name)
    model_file      = frame_cfg["model_file"]
    module          = importlib.import_module(f"models.{model_file}")

    model = module.build(
        model_cfg       = cfg["model"],
        frame_cfg       = frame_cfg,
        prev_params_dim = prev_params_dim,
    ).to(device)

    train_params = _resolve_frame_training(cfg, frame_cfg)
    opt          = build_optimizer(cfg, model, lr=train_params["learning_rate"])

    epochs       = cfg["training"]["epochs"]
    sched_type   = "plateau" if use_lr_scheduler else train_params["lr_scheduler"]
    scheduler    = build_lr_scheduler(opt, sched_type, epochs)

    grad_clip  = train_params["grad_clip"]
    cat_weight = cfg["training"].get("loss", {}).get("cat_weight", 1.0)
    n_mels     = cfg["features"]["n_mels"]

    train_loader, val_loader, _ = build_frame_dataloaders(cfg, frame_name)

    num_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_file}  |  Trainable params: {num_params_model:,}  "
          f"|  prev_params_dim: {prev_params_dim}")

    patience   = train_params["patience"]
    best_val   = float("inf")
    no_improve = 0

    save_best = cfg["training"]["checkpoint"].get("save_best", True)
    save_last = cfg["training"]["checkpoint"].get("save_last", True)

    cat_names: List[str] = list(frame_cfg.get("categorical_params", {}).keys())
    log_data: dict = {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": [],
                      "grad_norm": []}
    for name in cat_names:
        log_data[f"train_ce_{name}"] = []
        log_data[f"val_ce_{name}"]   = []

    print(f"\nTraining for {epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_m = _train_epoch(model, train_loader, opt, module.compute_loss,
                               frame_cfg, n_mels, cat_weight, device, grad_clip=grad_clip)
        val_m   = _eval_epoch(model,  val_loader,       module.compute_loss,
                               frame_cfg, n_mels, cat_weight, device)

        log_data["train_loss"].append(train_m["loss"])
        log_data["val_loss"].append(val_m["loss"])
        log_data["train_mse"].append(train_m["mse"])
        log_data["val_mse"].append(val_m["mse"])
        log_data["grad_norm"].append(train_m["grad_norm"])
        for name in cat_names:
            log_data[f"train_ce_{name}"].append(train_m.get(f"ce_{name}", 0.0))
            log_data[f"val_ce_{name}"].append(val_m.get(f"ce_{name}", 0.0))

        if scheduler is not None:
            if sched_type == "plateau":
                scheduler.step(val_m["loss"])
            else:
                scheduler.step()

        improved = val_m["loss"] < best_val
        if improved:
            best_val   = val_m["loss"]
            no_improve = 0
            if save_best:
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1

        cat_str = "  ".join(f"ce_{n}={val_m.get(f'ce_{n}', 0):.4f}" for n in cat_names)
        lr_now  = opt.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:04d}/{epochs}  "
            f"train={train_m['loss']:.6f}  val={val_m['loss']:.6f}  "
            f"mse={val_m['mse']:.6f}  grad_norm={train_m['grad_norm']:.4f}  lr={lr_now:.2e}"
            + (f"  {cat_str}" if cat_str else "")
            + ("  *" if improved else "")
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    if save_last:
        torch.save(model.state_dict(), ckpt_dir / "last.pt")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  |  Best val loss: {best_val:.6f}")

    log_data["training_time_seconds"] = round(elapsed, 1)
    log_data["training_time_human"]   = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s"

    log_data["epochs_run"]    = epoch
    log_data["best_epoch"]    = int(log_data["val_loss"].index(min(log_data["val_loss"]))) + 1
    log_data["best_val_loss"] = best_val
    log_data["frame_name"]    = frame_name
    log_data["dataset_size"]  = len(train_loader.dataset) + len(val_loader.dataset)
    log_data["hyperparameters"] = {
        "learning_rate": train_params["learning_rate"],
        "batch_size":    cfg["training"]["batch_size"],
        "dropout":       cfg["model"].get("dropout", 0.3),
        "patience":      patience,
        "lr_scheduler":  sched_type,
        "grad_clip":     grad_clip,
    }
    log_data["hardware"] = {
        "device": str(device),
    }

    torch.save({
        "model_cfg":       cfg["model"],
        "frame_cfg":       frame_cfg,
        "frame_name":      frame_name,
        "prev_params_dim": prev_params_dim,
        "param_names":     frame_cfg["params"],
        "state_dict":      model.state_dict(),
        "best_val_loss":   best_val,
    }, ckpt_dir / "best_full.pt")

    with open(log_dir / "train_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs:        {log_dir}")

    if cfg.get("export", {}).get("auto_export", False):
        print(f"\nAuto-exporting {frame_name} ONNX...")
        try:
            import sequenced_export_onnx
            sequenced_export_onnx.export_frame(cfg, frame_name,
                                               ckpt_dir / "best_full.pt",
                                               torch.device("cpu"))
        except Exception as e:
            print(f"ONNX export failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Exp### config.yaml")
    parser.add_argument("--frame",  required=True,
                        help="Frame name (e.g. 'filter') or 'all'")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-scheduler", action="store_true", default=False,
                        help="Force ReduceLROnPlateau, overriding config training.lr_scheduler")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg         = load_config(config_path)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["training"]["early_stopping"]["patience"] = args.patience
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    nn_root = Path(__file__).parent
    sys.path.insert(0, str(nn_root))

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()         else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    if args.frame == "all":
        frame_names = cfg["training"].get("frame_order") or [f["name"] for f in cfg["frames"]]
        for frame_name in frame_names:
            print(f"\n{'=' * 60}")
            print(f"=== Training frame: {frame_name} ===")
            print(f"{'=' * 60}")
            _train_frame(cfg, frame_name, device, use_lr_scheduler=args.lr_scheduler)
    else:
        _train_frame(cfg, args.frame, device, use_lr_scheduler=args.lr_scheduler)


if __name__ == "__main__":
    main()
