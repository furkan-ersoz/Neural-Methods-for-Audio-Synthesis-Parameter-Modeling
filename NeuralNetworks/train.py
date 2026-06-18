"""
train.py
Kullanim:
  python train.py --config synths/SynthV1/experiments/Exp001/config.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
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


def build_model(cfg: dict) -> nn.Module:
    model_class = cfg["model"]["class"].lower()          # "model001"
    module      = importlib.import_module(f"models.{model_class}")
    num_params  = len(cfg["dataset"]["params"])
    return module.build(cfg["model"], output_dim=num_params)


def build_loss(cfg: dict):
    loss_type    = cfg["training"].get("loss", "mse")
    loss_weights = cfg["training"].get("loss_weights", None)
    num_params   = len(cfg["dataset"]["params"])

    if loss_type == "mse":
        return nn.MSELoss()

    elif loss_type == "log_mse":
        # Log-space MSE — dusuk degerlerdeki hatalara daha fazla ceza
        eps = 1e-6
        def log_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return nn.functional.mse_loss(
                torch.log(pred.clamp(min=eps)),
                torch.log(target.clamp(min=eps))
            )
        return log_mse_loss

    elif loss_type == "weighted_mse":
        if loss_weights is None:
            loss_weights = [1.0] * num_params
        w = torch.tensor(loss_weights, dtype=torch.float32)

        def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            weights = w.to(pred.device)
            return (weights * (pred - target) ** 2).mean()

        return weighted_mse_loss

    else:
        raise ValueError(f"Unknown loss: {loss_type!r}. Choose: mse | log_mse | weighted_mse")


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    opt_type = cfg["training"].get("optimizer", "adam").lower()
    lr       = cfg["training"].get("learning_rate", 1e-3)

    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for features, params in loader:
        features = features.to(device)
        params   = params.to(device)

        optimizer.zero_grad()
        pred = model(features)
        loss = loss_fn(pred, params)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    for features, params in loader:
        features = features.to(device)
        params   = params.to(device)
        pred     = model(features)
        loss     = loss_fn(pred, params)
        total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Exp### config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg         = load_config(config_path)

    # ── Dizinler ──────────────────────────────────────────────────────────────
    exp_dir      = config_path.parent
    ckpt_dir     = exp_dir / "checkpoints"
    log_dir      = exp_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cuda")  if torch.cuda.is_available()         else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # dataset.py ile ayni dizinde olmali ya da PYTHONPATH'e eklenmeli
    import sys
    nn_root = Path(__file__).parent
    sys.path.insert(0, str(nn_root))

    from dataset import build_dataloaders
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model   = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    opt     = build_optimizer(cfg, model)

    num_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['class']}  |  Trainable params: {num_params_model:,}")

    # ── Early stopping ────────────────────────────────────────────────────────
    es_cfg    = cfg["training"].get("early_stopping", {})
    patience  = es_cfg.get("patience", 999)
    best_val  = float("inf")
    no_improve = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs    = cfg["training"]["epochs"]
    log_data  = {"train_loss": [], "val_loss": []}
    save_best = cfg["training"]["checkpoint"].get("save_best", True)
    save_last = cfg["training"]["checkpoint"].get("save_last", True)

    print(f"\nTraining for {epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, opt, device)
        val_loss   = eval_epoch(model,  val_loader,   loss_fn, device)

        log_data["train_loss"].append(train_loss)
        log_data["val_loss"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            no_improve = 0
            if save_best:
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1

        print(f"Epoch {epoch:04d}/{epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}"
              f"{'  *' if improved else ''}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    if save_last:
        torch.save(model.state_dict(), ckpt_dir / "last.pt")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  |  Best val loss: {best_val:.6f}")

    # ── Log kaydet ────────────────────────────────────────────────────────────
    with open(log_dir / "train_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # ── Model config kaydet (export icin lazim) ───────────────────────────────
    torch.save({
        "model_cfg":  cfg["model"],
        "output_dim": len(cfg["dataset"]["params"]),
        "param_names": cfg["dataset"]["params"],
        "state_dict": model.state_dict(),
        "best_val_loss": best_val,
    }, ckpt_dir / "best_full.pt")

    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs:        {log_dir}")

    # ── Auto ONNX export ──────────────────────────────────────────────────────
    if cfg.get("export", {}).get("auto_export", False):
        print("\nAuto-exporting ONNX...")
        try:
            import export_onnx
            export_onnx.run_export(cfg, ckpt_dir / "best_full.pt", torch.device("cpu"))
        except Exception as e:
            print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    main()