"""
sequenced_evaluate.py
Usage:
  python sequenced_evaluate.py --config synths/SynthV3/Experiments/Exp001/config.yaml
  python sequenced_evaluate.py --config ... --frame filter
  python sequenced_evaluate.py --config ... --frame door
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def get_device() -> torch.device:
    return (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Door evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_door_model(cfg: dict, ckpt_path: Path, device: torch.device):
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_file = cfg["door"].get("model_file", "model002")
    module     = importlib.import_module(f"models.{model_file}")
    model      = module.build(ckpt["model_cfg"], ckpt["door_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_door(model, loader, device: torch.device,
                  bypass_cols: List[str], lfo_classes: int) -> dict:
    all_bypass_pred   = []
    all_bypass_target = []
    all_lfo_pred      = []
    all_lfo_target    = []

    for mel, target in loader:
        mel    = mel.to(device)
        target = target.to(device)

        bypass_out, lfo_out = model(mel)               # sigmoid, softmax

        all_bypass_pred.append((bypass_out > 0.5).cpu().float().numpy())
        all_bypass_target.append(target[:, :len(bypass_cols)].cpu().numpy())
        all_lfo_pred.append(lfo_out.argmax(dim=-1).cpu().numpy())
        all_lfo_target.append(target[:, len(bypass_cols)].cpu().long().numpy())

    bypass_pred   = np.concatenate(all_bypass_pred)    # (N, n_bypass)
    bypass_target = np.concatenate(all_bypass_target)
    lfo_pred      = np.concatenate(all_lfo_pred)
    lfo_target    = np.concatenate(all_lfo_target)

    from sklearn.metrics import precision_score, recall_score, f1_score

    # Per-flag accuracy + precision/recall/F1
    bypass_accuracy:    Dict[str, float] = {}
    per_flag_precision: Dict[str, float] = {}
    per_flag_recall:    Dict[str, float] = {}
    per_flag_f1:        Dict[str, float] = {}
    for i, col in enumerate(bypass_cols):
        acc = float(np.mean(bypass_pred[:, i] == bypass_target[:, i]))
        bypass_accuracy[col]    = acc
        per_flag_precision[col] = float(precision_score(bypass_target[:, i], bypass_pred[:, i], zero_division=0))
        per_flag_recall[col]    = float(recall_score(bypass_target[:, i], bypass_pred[:, i], zero_division=0))
        per_flag_f1[col]        = float(f1_score(bypass_target[:, i], bypass_pred[:, i], zero_division=0))

    overall_bypass_acc = float(np.mean(list(bypass_accuracy.values())))
    lfo_count_accuracy = float(np.mean(lfo_pred == lfo_target))
    lfo_macro_f1       = float(f1_score(lfo_target, lfo_pred, average="macro",    zero_division=0))
    lfo_weighted_f1    = float(f1_score(lfo_target, lfo_pred, average="weighted", zero_division=0))

    # LFO confusion matrix
    cm = np.zeros((lfo_classes, lfo_classes), dtype=int)
    for t, p in zip(lfo_target, lfo_pred):
        cm[int(t), int(p)] += 1

    return {
        "bypass_accuracy":        bypass_accuracy,
        "per_flag_precision":     per_flag_precision,
        "per_flag_recall":        per_flag_recall,
        "per_flag_f1":            per_flag_f1,
        "overall_bypass_accuracy": overall_bypass_acc,
        "lfo_count_accuracy":     lfo_count_accuracy,
        "lfo_macro_f1":           lfo_macro_f1,
        "lfo_weighted_f1":        lfo_weighted_f1,
        "lfo_confusion_matrix":   cm.tolist(),
    }


def print_door_results(results: dict, bypass_cols: List[str]) -> None:
    print("\n=== Door Classifier ===")
    for col, acc in results["bypass_accuracy"].items():
        print(f"  {col} accuracy: {acc * 100:.1f}%")
    print(f"  Overall bypass accuracy : {results['overall_bypass_accuracy'] * 100:.1f}%")
    print(f"  LFO count accuracy      : {results['lfo_count_accuracy'] * 100:.1f}%")

    cm = np.array(results["lfo_confusion_matrix"])
    n  = cm.shape[0]
    print("\n  LFO count confusion matrix (rows=true, cols=pred):")
    header = "      " + "  ".join(f"{i:3d}" for i in range(n))
    print(f"  {header}")
    print(f"  {'─' * (6 + n * 5)}")
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:3d}" for v in row)
        print(f"  {i:3d} | {cells}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Frame predictor evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_frame_model(ckpt_path: Path, device: torch.device):
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    module = importlib.import_module(f"models.{ckpt['frame_cfg']['model_file']}")
    model  = module.build(ckpt["model_cfg"], ckpt["frame_cfg"],
                          ckpt["prev_params_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_frame(model, loader, device: torch.device,
                   frame_cfg: dict, n_mels: int) -> dict:
    from models.model011 import continuous_predictions

    cat_names  = list(frame_cfg.get("categorical_params", {}).keys())
    all_params = list(frame_cfg["params"])
    cont_names = [p for p in all_params if p not in cat_names]

    all_cont_pred   = []
    all_cont_target = []
    cat_pred:   Dict[str, List] = {n: [] for n in cat_names}
    cat_target: Dict[str, List] = {n: [] for n in cat_names}
    total_loss = 0.0
    n_batches  = 0

    for mel, prev_params, target in loader:
        mel         = mel.to(device)
        prev_params = prev_params.to(device)
        target      = target.to(device)

        # Split mel channels
        mel_full = mel[:, :n_mels, :].unsqueeze(1)
        if mel.shape[1] > n_mels:
            mel_prev = mel[:, n_mels:, :].unsqueeze(1)
        else:
            mel_prev = torch.zeros_like(mel_full)

        cont_out, cat_outs = model(mel_full, mel_prev, prev_params)

        # Split target
        cont_indices = [i for i, p in enumerate(all_params) if p not in cat_names]
        cat_indices  = {p: i for i, p in enumerate(all_params) if p in cat_names}

        if cont_indices:
            cont_t = target[:, cont_indices]
            all_cont_pred.append(continuous_predictions(cont_out).cpu().numpy())
            all_cont_target.append(cont_t.cpu().numpy())

        for name in cat_names:
            idx    = cat_indices[name]
            t      = target[:, idx].long()
            logits = cat_outs[name]
            cat_pred[name].append(logits.argmax(dim=-1).cpu().numpy())
            cat_target[name].append(t.cpu().numpy())

        # Val loss (same as training)
        import torch.nn.functional as F
        loss = torch.tensor(0.0, device=device)
        if cont_indices:
            cont_t = target[:, cont_indices]
            loss   = loss + F.mse_loss(continuous_predictions(cont_out), cont_t)
        for name, logits in cat_outs.items():
            t    = target[:, cat_indices[name]].long()
            loss = loss + F.cross_entropy(logits, t)
        total_loss += loss.item()
        n_batches  += 1

    results: dict = {}
    results["val_loss"] = total_loss / max(n_batches, 1)

    if all_cont_pred:
        cp = np.concatenate(all_cont_pred)
        ct = np.concatenate(all_cont_target)
        mse_per = {cont_names[i]: float(np.mean((cp[:, i] - ct[:, i]) ** 2))
                   for i in range(len(cont_names))}
        mae_per = {cont_names[i]: float(np.mean(np.abs(cp[:, i] - ct[:, i])))
                   for i in range(len(cont_names))}
        results["mse_per_param"] = mse_per
        results["mae_per_param"] = mae_per
        results["overall_mse"]   = float(np.mean((cp - ct) ** 2))
        results["overall_mae"]   = float(np.mean(np.abs(cp - ct)))

    for name in cat_names:
        pp = np.concatenate(cat_pred[name])
        pt = np.concatenate(cat_target[name])
        results[f"{name}_accuracy"] = float(np.mean(pp == pt))

        import torch.nn.functional as F
        logit_list = []
        target_list = []
        # recompute CE from stored predictions is non-trivial; store accuracy only
        # CE is already folded into val_loss

    return results


def print_frame_results(results: dict, frame_name: str, frame_cfg: dict) -> None:
    cat_names = list(frame_cfg.get("categorical_params", {}).keys())
    all_params = list(frame_cfg["params"])
    cont_names = [p for p in all_params if p not in cat_names]

    print(f"\n=== Frame: {frame_name} ===")
    print(f"  Val loss: {results['val_loss']:.6f}")

    if "mse_per_param" in results:
        print(f"  {'Param':<20}  {'MSE':>10}  {'MAE':>10}")
        print(f"  {'─' * 44}")
        for p in cont_names:
            print(f"  {p:<20}  {results['mse_per_param'][p]:>10.6f}  "
                  f"{results['mae_per_param'][p]:>10.6f}")
        print(f"  {'overall':<20}  {results['overall_mse']:>10.6f}  "
              f"{results['overall_mae']:>10.6f}")

    for name in cat_names:
        acc = results.get(f"{name}_accuracy", float("nan"))
        print(f"  {name} accuracy: {acc * 100:.1f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: Dict[str, dict]) -> None:
    print("=== Evaluation Summary ===")
    print(f"{'Component':<16} | {'Val Loss':>8} | Notes")
    print(f"{'─' * 16}-|-{'─' * 8}-|-{'─' * 35}")
    for name, res in all_results.items():
        if name == "door":
            loss  = res.get("val_loss", float("nan"))
            b_acc = res["overall_bypass_accuracy"] * 100
            l_acc = res["lfo_count_accuracy"] * 100
            notes = f"bypass_acc={b_acc:.1f}% lfo_acc={l_acc:.1f}%"
        else:
            loss  = res.get("val_loss", float("nan"))
            notes_parts = []
            if "overall_mse" in res:
                notes_parts.append(f"mse={res['overall_mse']:.3f}")
            if "overall_mae" in res:
                notes_parts.append(f"mae={res['overall_mae']:.3f}")
            for k, v in res.items():
                if k.endswith("_accuracy"):
                    param = k[:-9]
                    notes_parts.append(f"{param}_acc={v * 100:.1f}%")
            notes = " ".join(notes_parts)
        loss_str = f"{loss:.4f}" if not np.isnan(loss) else "  n/a  "
        print(f"{name:<16} | {loss_str:>8} | {notes}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Exp### config.yaml")
    parser.add_argument("--frame",  default=None,
                        help="door | <frame_name> | omit for all")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg         = load_config(config_path)
    exp_dir     = config_path.parent

    nn_root = Path(__file__).parent
    sys.path.insert(0, str(nn_root))
    from sequenced_dataset import build_door_dataloaders, build_frame_dataloaders

    device  = get_device()
    n_mels  = cfg["features"]["n_mels"]

    door_cfg    = cfg.get("door", {})
    bypass_cols = door_cfg.get("bypass_columns", [
        "bypass_osc_b", "bypass_noise", "bypass_filter",
        "bypass_reverb", "bypass_delay",
    ])
    lfo_classes = int(door_cfg.get("lfo_count_classes", 5))

    all_results: Dict[str, dict] = {}

    evaluate_door_flag   = args.frame is None or args.frame == "door"
    evaluate_frames_flag = args.frame is None or args.frame not in ("door",)

    # Determine which frame names to evaluate
    if args.frame is None:
        frame_names = [f["name"] for f in cfg.get("frames", [])]
    elif args.frame == "door":
        frame_names = []
    else:
        frame_names = [args.frame]

    # ── Door ──────────────────────────────────────────────────────────────────
    if evaluate_door_flag:
        ckpt_path = exp_dir / "checkpoints" / "door" / "best_full.pt"
        if not ckpt_path.exists():
            print(f"WARNING: door checkpoint not found ({ckpt_path}), skipping.")
        else:
            print(f"Loading door checkpoint: {ckpt_path}")
            model, ckpt = load_door_model(cfg, ckpt_path, device)
            _, val_loader, test_loader = build_door_dataloaders(cfg)
            val_metrics  = evaluate_door(model, val_loader,  device, bypass_cols, lfo_classes)
            test_metrics = evaluate_door(model, test_loader, device, bypass_cols, lfo_classes)

            # Latency: CPU, batch_size=1, 100 warmup + 500 timed runs
            lat_model = model.to(torch.device("cpu")).eval()
            for mel_lat, _ in test_loader:
                mel_lat = mel_lat[:1].to(torch.device("cpu"))
                break
            with torch.no_grad():
                for _ in range(100):
                    lat_model(mel_lat)
                lat_times = []
                for _ in range(500):
                    t0 = time.perf_counter()
                    lat_model(mel_lat)
                    lat_times.append((time.perf_counter() - t0) * 1000)

            val_loss = ckpt.get("best_val_loss", float("nan"))
            train_log_path = exp_dir / "logs" / "door" / "train_log.json"
            training_meta = {}
            if train_log_path.exists():
                with open(train_log_path) as f:
                    tlog = json.load(f)
                training_meta = {
                    "epochs_run":      tlog.get("epochs_run"),
                    "best_epoch":      tlog.get("best_epoch"),
                    "best_val_loss":   tlog.get("best_val_loss"),
                    "dataset_size":    tlog.get("dataset_size"),
                    "hyperparameters": tlog.get("hyperparameters"),
                    "hardware":        tlog.get("hardware"),
                }
            results = {
                # top-level keys kept for backwards compatibility
                "val_loss":                val_loss,
                "bypass_accuracy":         val_metrics["bypass_accuracy"],
                "overall_bypass_accuracy": val_metrics["overall_bypass_accuracy"],
                "lfo_count_accuracy":      val_metrics["lfo_count_accuracy"],
                "lfo_confusion_matrix":    val_metrics["lfo_confusion_matrix"],
                "training": training_meta,
                "val": {
                    "loss":               val_loss,
                    "bypass_accuracy":    val_metrics["bypass_accuracy"],
                    "per_flag_precision": val_metrics["per_flag_precision"],
                    "per_flag_recall":    val_metrics["per_flag_recall"],
                    "per_flag_f1":        val_metrics["per_flag_f1"],
                    "lfo_count_accuracy": val_metrics["lfo_count_accuracy"],
                    "lfo_macro_f1":       val_metrics["lfo_macro_f1"],
                    "lfo_weighted_f1":    val_metrics["lfo_weighted_f1"],
                    "lfo_confusion_matrix": val_metrics["lfo_confusion_matrix"],
                },
                "test": {
                    "bypass_accuracy":    test_metrics["bypass_accuracy"],
                    "per_flag_precision": test_metrics["per_flag_precision"],
                    "per_flag_recall":    test_metrics["per_flag_recall"],
                    "per_flag_f1":        test_metrics["per_flag_f1"],
                    "lfo_count_accuracy": test_metrics["lfo_count_accuracy"],
                    "lfo_macro_f1":       test_metrics["lfo_macro_f1"],
                    "lfo_weighted_f1":    test_metrics["lfo_weighted_f1"],
                    "lfo_confusion_matrix": test_metrics["lfo_confusion_matrix"],
                },
                "latency": {
                    "mean_ms": float(np.mean(lat_times)),
                    "std_ms":  float(np.std(lat_times)),
                    "device":  "cpu",
                    "runs":    500,
                },
            }
            print_door_results(results, bypass_cols)
            all_results["door"] = results

    # ── Frames ────────────────────────────────────────────────────────────────
    for frame_name in frame_names:
        ckpt_path = exp_dir / "checkpoints" / frame_name / "best_full.pt"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint not found for frame '{frame_name}' "
                  f"({ckpt_path}), skipping.")
            continue

        print(f"Loading frame checkpoint: {ckpt_path}")
        model, ckpt = load_frame_model(ckpt_path, device)
        _, val_loader, _ = build_frame_dataloaders(cfg, frame_name)

        frame_cfg = ckpt["frame_cfg"]
        results   = evaluate_frame(model, val_loader, device, frame_cfg, n_mels)
        if "val_loss" not in results or np.isnan(results.get("val_loss", float("nan"))):
            results["val_loss"] = ckpt.get("best_val_loss", float("nan"))
        train_log_path = exp_dir / "logs" / frame_name / "train_log.json"
        training_meta = {}
        if train_log_path.exists():
            with open(train_log_path) as f:
                tlog = json.load(f)
            training_meta = {
                "epochs_run":      tlog.get("epochs_run"),
                "best_epoch":      tlog.get("best_epoch"),
                "best_val_loss":   tlog.get("best_val_loss"),
                "dataset_size":    tlog.get("dataset_size"),
                "hyperparameters": tlog.get("hyperparameters"),
                "hardware":        tlog.get("hardware"),
            }
        results["training"] = training_meta
        print_frame_results(results, frame_name, frame_cfg)
        all_results.setdefault("frames", {})[frame_name] = results

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print_summary(all_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "eval_results.json"

    # Convert numpy/nan to JSON-safe types
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    import hashlib
    import datetime

    meta = {
        "experiment":   cfg["experiment"]["name"],
        "dataset_path": cfg["dataset"]["path"],
        "config_hash":  hashlib.md5(open(args.config, "rb").read()).hexdigest()[:8],
        "evaluated_at": datetime.datetime.now().isoformat(),
        "frame":        args.frame,
    }

    new_results = _jsonify({"meta": meta, **all_results})

    results_path = out_path
    existing = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            existing = json.load(f)

    existing["meta"] = new_results["meta"]
    if "door" in new_results:
        existing["door"] = new_results["door"]
    if "frames" in new_results:
        existing.setdefault("frames", {}).update(new_results.get("frames", {}))

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"Evaluation results saved: {results_path}")


if __name__ == "__main__":
    main()
