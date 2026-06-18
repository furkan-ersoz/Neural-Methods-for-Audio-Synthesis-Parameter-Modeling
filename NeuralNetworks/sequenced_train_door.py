"""
sequenced_train_door.py
Phase 1 — Door Classifier (model002) training.

Usage:
  python sequenced_train_door.py \
    --config synths/SynthV3/Experiments/Exp001/config.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml


# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_exp_dir"]     = str(config_path.parent)
    cfg["_config_path"] = str(config_path)
    return cfg


def _door_model_module(cfg: dict):
    model_file = cfg["door"].get("model_file", "model002")
    return importlib.import_module(f"models.{model_file}")


def build_model(cfg: dict) -> nn.Module:
    module = _door_model_module(cfg)
    return module.build(cfg["model"], cfg["door"])


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    opt_type = cfg["training"].get("optimizer", "adam").lower()
    lr       = cfg["training"].get("learning_rate", 1e-3)
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


def _get_loss_weights(cfg: dict) -> dict | None:
    return cfg["door"].get("loss_weights", None)


# ─────────────────────────────────────────────────────────────────────────────
def _split_target(target: torch.Tensor, n_bypass: int):
    """target (B, n_bypass+1) → bypass_target (B, n_bypass), lfo_target (B,)"""
    bypass_target = target[:, :n_bypass].float()
    lfo_target    = target[:, n_bypass].long()
    return bypass_target, lfo_target


def _bypass_accuracy(bypass_pred_sig: torch.Tensor,
                     bypass_target: torch.Tensor) -> float:
    """Mean per-flag accuracy using 0.5 threshold."""
    pred_binary = (bypass_pred_sig >= 0.5).float()
    return (pred_binary == bypass_target).float().mean().item()


def _lfo_accuracy(lfo_logits: torch.Tensor, lfo_target: torch.Tensor) -> float:
    return (lfo_logits.argmax(dim=-1) == lfo_target).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
def _train_epoch(model, loader, optimizer, loss_weights, device, n_bypass,
                 compute_loss, lfo_class_weights=None, grad_clip=None) -> dict:
    model.train()
    total_loss   = 0.0
    total_bypass = 0.0
    total_lfo    = 0.0
    total_grad_norm = 0.0
    n_batches    = 0

    for mel, target in loader:
        mel    = mel.to(device)
        target = target.to(device)

        bypass_target, lfo_target = _split_target(target, n_bypass)

        optimizer.zero_grad()
        bypass_logits, lfo_logits = model.forward_logits(mel)
        loss = compute_loss(bypass_logits, lfo_logits,
                            bypass_target, lfo_target, loss_weights,
                            lfo_class_weights)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip if grad_clip else float("inf")
        )
        total_grad_norm += grad_norm.item()
        n_batches += 1
        optimizer.step()

        with torch.no_grad():
            bypass_sig = torch.sigmoid(bypass_logits)
            total_bypass += _bypass_accuracy(bypass_sig, bypass_target) * mel.size(0)
            total_lfo    += _lfo_accuracy(lfo_logits, lfo_target)      * mel.size(0)
            total_loss   += loss.item()                                 * mel.size(0)

    n = len(loader.dataset)
    return {"loss": total_loss / n, "bypass_acc": total_bypass / n, "lfo_acc": total_lfo / n,
            "grad_norm": total_grad_norm / max(n_batches, 1)}


@torch.no_grad()
def _eval_epoch(model, loader, loss_weights, device, n_bypass,
                compute_loss, lfo_class_weights=None) -> dict:
    model.eval()
    total_loss   = 0.0
    total_bypass = 0.0
    total_lfo    = 0.0

    for mel, target in loader:
        mel    = mel.to(device)
        target = target.to(device)

        bypass_target, lfo_target = _split_target(target, n_bypass)

        bypass_logits, lfo_logits = model.forward_logits(mel)
        loss = compute_loss(bypass_logits, lfo_logits,
                            bypass_target, lfo_target, loss_weights,
                            lfo_class_weights)

        bypass_sig = torch.sigmoid(bypass_logits)
        total_bypass += _bypass_accuracy(bypass_sig, bypass_target) * mel.size(0)
        total_lfo    += _lfo_accuracy(lfo_logits, lfo_target)      * mel.size(0)
        total_loss   += loss.item()                                 * mel.size(0)

    n = len(loader.dataset)
    if n == 0:
        return {"loss": 0.0, "bypass_acc": 0.0, "lfo_acc": 0.0}
    return {"loss": total_loss / n, "bypass_acc": total_bypass / n, "lfo_acc": total_lfo / n}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Exp### config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-scheduler", action="store_true", default=False)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg         = load_config(config_path)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["training"]["early_stopping"]["patience"] = args.patience
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    # ── Directories ───────────────────────────────────────────────────────────
    exp_dir  = Path(cfg["_exp_dir"])
    ckpt_dir = exp_dir / "checkpoints" / "door"
    log_dir  = exp_dir / "logs" / "door"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()         else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Path setup ────────────────────────────────────────────────────────────
    nn_root = Path(__file__).parent
    sys.path.insert(0, str(nn_root))

    from sequenced_dataset import build_door_dataloaders
    train_loader, val_loader, _ = build_door_dataloaders(cfg)

    # ── LFO class weights (inverse frequency) ────────────────────────────────
    lfo_count_classes = int(cfg["door"]["lfo_count_classes"])
    use_lfo_class_weights = cfg["door"].get("use_lfo_class_weights", True)

    from collections import Counter
    raw_rows = train_loader.dataset.dataset.rows
    lfo_counts = Counter(int(r["active_lfo_count"]) for r in raw_rows)
    n_total    = sum(lfo_counts.values())

    lfo_class_weights = None
    if use_lfo_class_weights:
        lfo_class_weights = torch.tensor(
            [n_total / (lfo_count_classes * lfo_counts.get(i, 1)) for i in range(lfo_count_classes)],
            dtype=torch.float32,
            device=device,
        )
        print(f"LFO class weights: {lfo_class_weights.tolist()}")

    # ── Model ─────────────────────────────────────────────────────────────────
    door_module  = _door_model_module(cfg)
    compute_loss = door_module.compute_loss
    model        = build_model(cfg).to(device)
    opt          = build_optimizer(cfg, model)

    epochs       = cfg["training"]["epochs"]
    sched_type   = "plateau" if args.lr_scheduler else cfg["training"].get("lr_scheduler", "cosine")
    scheduler    = build_lr_scheduler(opt, sched_type, epochs)
    grad_clip    = cfg["training"].get("grad_clip")

    loss_weights = _get_loss_weights(cfg)
    n_bypass     = len(cfg["door"]["bypass_columns"])

    model_file = cfg["door"].get("model_file", "model002")
    num_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_file}  |  Trainable params: {num_params_model:,}")
    print(f"Bypass flags: {n_bypass}  |  LFO count classes: {cfg['door']['lfo_count_classes']}")

    # ── Early stopping ────────────────────────────────────────────────────────
    es_cfg     = cfg["training"].get("early_stopping", {})
    patience   = es_cfg.get("patience", 999)
    best_val   = float("inf")
    no_improve = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    save_best = cfg["training"]["checkpoint"].get("save_best", True)
    save_last = cfg["training"]["checkpoint"].get("save_last", True)

    log_data = {
        "train_loss": [], "val_loss": [],
        "train_bypass_acc": [], "val_bypass_acc": [],
        "train_lfo_acc": [],   "val_lfo_acc": [],
        "grad_norm": [],
    }

    print(f"\nTraining for {epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_m = _train_epoch(model, train_loader, opt, loss_weights, device, n_bypass,
                               compute_loss, lfo_class_weights, grad_clip=grad_clip)
        val_m   = _eval_epoch(model,  val_loader,       loss_weights, device, n_bypass,
                              compute_loss, lfo_class_weights)

        log_data["train_loss"].append(train_m["loss"])
        log_data["val_loss"].append(val_m["loss"])
        log_data["train_bypass_acc"].append(train_m["bypass_acc"])
        log_data["val_bypass_acc"].append(val_m["bypass_acc"])
        log_data["train_lfo_acc"].append(train_m["lfo_acc"])
        log_data["val_lfo_acc"].append(val_m["lfo_acc"])
        log_data["grad_norm"].append(train_m["grad_norm"])

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

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:04d}/{epochs}  "
            f"train={train_m['loss']:.6f}  val={val_m['loss']:.6f}  "
            f"bypass_acc={val_m['bypass_acc']:.4f}  lfo_acc={val_m['lfo_acc']:.4f}  "
            f"grad_norm={train_m['grad_norm']:.4f}  lr={lr_now:.2e}"
            f"{'  *' if improved else ''}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    if save_last:
        torch.save(model.state_dict(), ckpt_dir / "last.pt")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  |  Best val loss: {best_val:.6f}")

    log_data["epochs_run"]    = epoch
    log_data["best_epoch"]    = int(log_data["val_loss"].index(min(log_data["val_loss"]))) + 1
    log_data["best_val_loss"] = best_val
    log_data["dataset_size"]  = len(train_loader.dataset) + len(val_loader.dataset)
    log_data["hyperparameters"] = {
        "learning_rate": cfg["training"]["learning_rate"],
        "batch_size":    cfg["training"]["batch_size"],
        "dropout":       cfg["model"].get("dropout", 0.3),
        "patience":      cfg["training"]["early_stopping"].get("patience", 20),
        "lr_scheduler":  sched_type,
        "grad_clip":     grad_clip,
    }
    log_data["hardware"] = {
        "device": str(device),
    }

    # ── Full checkpoint ───────────────────────────────────────────────────────
    torch.save({
        "model_cfg": cfg["model"],
        "door_cfg":  cfg["door"],
        "state_dict": model.state_dict(),
        "best_val_loss": best_val,
    }, ckpt_dir / "best_full.pt")

    # ── Log ───────────────────────────────────────────────────────────────────
    with open(log_dir / "train_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs:        {log_dir}")

    # ── Auto ONNX export ──────────────────────────────────────────────────────
    if cfg.get("export", {}).get("auto_export", False):
        print("\nAuto-exporting door ONNX...")
        try:
            import sequenced_export_onnx
            sequenced_export_onnx.export_door(cfg, ckpt_dir / "best_full.pt",
                                              torch.device("cpu"))
        except Exception as e:
            print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    main()
