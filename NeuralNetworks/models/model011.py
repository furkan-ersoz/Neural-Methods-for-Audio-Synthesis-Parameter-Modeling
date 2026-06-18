"""
models/model011.py — Base architecture v2 (pool/temporal/in_channels fixes)

Changes vs model003-010:
  - pool_mode "freq"  : AdaptiveAvgPool2d((F_out, 1)) — keeps frequency axis
  - pool_mode "stats" : mean+std over (freq, time) -> 2*C_out dims
  - temporal "gru"    : single-layer GRU over the CNN time axis,
                        last hidden state appended to the pooled features
  - in_channels computed from model_cfg/frame_cfg (stereo-ready), not hardcoded

build()/compute_loss() signatures match model003-010.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
class CNNBlock2d(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d(2×2)"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


# ─────────────────────────────────────────────────────────────────────────────
class FramePredictor(nn.Module):
    """
    Base frame predictor v2.

    Parameters
    ----------
    n_mels             : int            — mel bins (frequency axis)
    prev_params_dim    : int            — D: total param count from preceding frames
    continuous_params  : List[str]      — names of continuous outputs
    categorical_params : Dict[str, List[str]] — {param_name: [class_labels]}
    in_channels        : int            — CNN input channels (mel_full + mel_prev, *audio_channels)
    cnn_channels       : List[int]      — out channels per CNN block
    cond_dim           : int            — conditioning projection size
    fc_hidden          : int            — fused FC hidden size
    dropout            : float
    pool_mode          : "freq" | "stats"
    pool_f_out         : int | None     — F_out for pool_mode="freq" (default n_mels // 4)
    temporal           : "none" | "gru"
    """

    def __init__(
        self,
        n_mels:             int,
        prev_params_dim:    int,
        continuous_params:  List[str],
        categorical_params: Dict[str, List[str]],
        in_channels:        int       = 2,
        cnn_channels:       List[int] = (32, 64, 128),
        cond_dim:           int       = 64,
        fc_hidden:          int       = 256,
        dropout:            float     = 0.3,
        pool_mode:          str       = "stats",
        pool_f_out:         Optional[int] = None,
        temporal:           str       = "none",
        continuous_mode:    str       = "regression",
        n_bins:             int       = 32,
    ) -> None:
        super().__init__()

        if pool_mode not in ("freq", "stats"):
            raise ValueError(f"pool_mode must be 'freq' or 'stats', got {pool_mode!r}")
        if temporal not in ("none", "gru"):
            raise ValueError(f"temporal must be 'none' or 'gru', got {temporal!r}")
        if continuous_mode not in ("regression", "classification"):
            raise ValueError(
                f"continuous_mode must be 'regression' or 'classification', got {continuous_mode!r}"
            )

        self.continuous_params  = continuous_params
        self.categorical_params = categorical_params
        self.prev_params_dim    = prev_params_dim
        self.n_continuous       = len(continuous_params)
        self.pool_mode          = pool_mode
        self.temporal           = temporal
        self.continuous_mode    = continuous_mode
        self.n_bins             = n_bins
        # Loss-time flag, set externally by build(); not used inside forward().
        self.per_param_loss_weight = False

        # ── CNN stack: (B, in_channels, n_mels, T) → (B, cnn_ch[-1], H', W') ──
        cnn_layers: List[nn.Module] = []
        in_ch = in_channels
        for out_ch in cnn_channels:
            cnn_layers.append(CNNBlock2d(in_ch, out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        cnn_out_ch = cnn_channels[-1]

        # ── Pooling head ────────────────────────────────────────────────────
        if pool_mode == "freq":
            f_out = pool_f_out if pool_f_out is not None else max(1, n_mels // 4)
            self.pool_f_out = f_out
            # H' after the CNN stack (each block halves the freq axis via MaxPool2d(2))
            freq_in = max(1, n_mels // (2 ** len(cnn_channels)))
            # Time is collapsed via mean (ONNX-safe under dynamic axes); the
            # freq axis (static size freq_in) is projected to f_out via Linear
            # -- AdaptiveAvgPool2d((f_out,1)) is not ONNX-exportable when f_out
            # does not evenly divide freq_in (e.g. upsampling 16 -> 32).
            self.freq_proj = nn.Linear(freq_in, f_out)
            pool_dim = cnn_out_ch * f_out
        else:  # "stats"
            self.pool_f_out = None
            self.freq_proj = None
            pool_dim = cnn_out_ch * 2  # mean + std

        # ── Temporal head ───────────────────────────────────────────────────
        if temporal == "gru":
            self.temporal_gru = nn.GRU(
                input_size=cnn_out_ch, hidden_size=cnn_out_ch,
                num_layers=1, batch_first=True,
            )
            gru_dim = cnn_out_ch
        else:
            self.temporal_gru = None
            gru_dim = 0

        # ── Conditioning projection ─────────────────────────────────────────
        self.has_cond = prev_params_dim > 0
        if self.has_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(prev_params_dim, cond_dim),
                nn.ReLU(inplace=True),
            )
        self.cond_dim = cond_dim

        # ── Fusion FC ──────────────────────────────────────────────────────
        fused_dim = pool_dim + gru_dim + cond_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Continuous head ───────────────────────────────────────────────
        if self.n_continuous == 0:
            self.continuous_head = None
        elif self.continuous_mode == "classification":
            # Per-param K-way logits, flattened: (B, n_continuous * n_bins).
            # Reshaped to (B, n_continuous, n_bins) in forward(). Exported to
            # ONNX as raw logits; softmax+weighted-bin-center happens outside.
            self.continuous_head = nn.Linear(fc_hidden, self.n_continuous * self.n_bins)
        else:
            self.continuous_head = nn.Sequential(
                nn.Linear(fc_hidden, self.n_continuous),
                nn.Sigmoid(),
            )

        # ── Categorical heads (one per param) ───────────────────────────────
        self.cat_heads = nn.ModuleDict({
            name: nn.Linear(fc_hidden, len(classes))
            for name, classes in categorical_params.items()
        })

    # ── Pooling helper ───────────────────────────────────────────────────────
    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H', T') -> (B, pool_dim)"""
        if self.pool_mode == "freq":
            # Collapse time first (ReduceMean, dynamic-axis safe), then
            # project the frequency axis (static size) to f_out.
            x_t = x.mean(dim=3)                  # (B, C, H')
            return self.freq_proj(x_t).flatten(1)  # (B, C * f_out)

        # "stats": collapse frequency by averaging, then mean+std over time
        x_t = x.mean(dim=2)                       # (B, C, T')
        mean = x_t.mean(dim=2)                    # (B, C)
        std  = x_t.std(dim=2, unbiased=False)     # (B, C)
        return torch.cat([mean, std], dim=1)      # (B, 2C)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        mel_full:    torch.Tensor,  # (B, C_in/2, n_mels, T)
        mel_prev:    torch.Tensor,  # (B, C_in/2, n_mels, T) — zero if no cheat sheet
        prev_params: torch.Tensor,  # (B, D)                  — zero if first frame
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        continuous_out   : (B, n_continuous) Sigmoid outputs (mode="regression"),
                           or (B, n_continuous, n_bins) raw logits (mode="classification")
        categorical_outs : {param_name: (B, n_classes)} — raw logits
        """
        B = mel_full.size(0)

        # CNN branch
        x = torch.cat([mel_full, mel_prev], dim=1)  # (B, C_in, n_mels, T)
        x = self.cnn(x)                              # (B, cnn_ch[-1], H', T')

        pooled = self._pool(x)                       # (B, pool_dim)

        if self.temporal_gru is not None:
            x_t = x.mean(dim=2)                       # (B, C, T') — collapse freq
            x_t = x_t.transpose(1, 2)                 # (B, T', C)
            _, h = self.temporal_gru(x_t)             # h: (1, B, C)
            pooled = torch.cat([pooled, h.squeeze(0)], dim=1)

        # Conditioning branch
        if self.has_cond:
            cond = self.cond_proj(prev_params)       # (B, cond_dim)
        else:
            cond = torch.zeros(B, self.cond_dim, device=mel_full.device)

        # Fuse
        fused = self.fusion(torch.cat([pooled, cond], dim=1))  # (B, fc_hidden)

        # Heads
        if self.continuous_head is None:
            continuous_out = torch.zeros(B, 0, device=mel_full.device)
        elif self.continuous_mode == "classification":
            continuous_out = self.continuous_head(fused).view(B, self.n_continuous, self.n_bins)
        else:
            continuous_out = self.continuous_head(fused)
        categorical_outs = {
            name: head(fused) for name, head in self.cat_heads.items()
        }

        return continuous_out, categorical_outs

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(
    continuous_pred:     torch.Tensor,
    categorical_preds:   Dict[str, torch.Tensor],
    continuous_target:   torch.Tensor,
    categorical_targets: Dict[str, torch.Tensor],
    cat_weight: float = 1.0,
    per_param_loss_weight: bool = False,
) -> torch.Tensor:
    """
    continuous loss + cat_weight * sum(CrossEntropy(each categorical))

    continuous_pred    : (B, n_continuous) Sigmoid outputs (regression mode), or
                         (B, n_continuous, n_bins) raw logits (classification mode)
    categorical_preds  : {name: (B, n_classes)} — raw logits
    continuous_target  : (B, n_continuous) — float [0, 1]
    categorical_targets: {name: (B,)} — integer class indices (long)

    Continuous mode is inferred from continuous_pred.dim():
      - dim == 3 -> classification: per-param K-way cross-entropy against the
        bin containing continuous_target.
      - dim == 2 -> regression: MSE. If per_param_loss_weight, each param's
        MSE (averaged over the batch) is weighted equally before averaging
        across params.
    """
    loss = torch.tensor(0.0, device=continuous_pred.device)

    if continuous_pred.numel() > 0:
        if continuous_pred.dim() == 3:
            n_bins = continuous_pred.shape[-1]
            target_bins = (continuous_target * n_bins).long().clamp(0, n_bins - 1)
            loss = loss + F.cross_entropy(continuous_pred.transpose(1, 2), target_bins)
        elif per_param_loss_weight:
            per_param_mse = (continuous_pred - continuous_target).pow(2).mean(dim=0)
            loss = loss + per_param_mse.mean()
        else:
            loss = loss + F.mse_loss(continuous_pred, continuous_target)

    for name, logits in categorical_preds.items():
        target = categorical_targets[name].long()
        loss = loss + cat_weight * F.cross_entropy(logits, target)

    return loss


# ─────────────────────────────────────────────────────────────────────────────
def continuous_predictions(continuous_pred: torch.Tensor) -> torch.Tensor:
    """
    Final [0, 1] continuous predictions, regardless of continuous_mode.

    regression    : continuous_pred is already (B, n_continuous) in [0, 1]
    classification: continuous_pred is (B, n_continuous, n_bins) raw logits;
                     softmax-weighted bin centers -> (B, n_continuous)
    """
    if continuous_pred.dim() == 3:
        n_bins = continuous_pred.shape[-1]
        centers = (torch.arange(n_bins, device=continuous_pred.device, dtype=continuous_pred.dtype) + 0.5) / n_bins
        probs = F.softmax(continuous_pred, dim=-1)
        return (probs * centers).sum(dim=-1)
    return continuous_pred


def continuous_metrics(continuous_pred: torch.Tensor, continuous_target: torch.Tensor) -> Dict[str, float]:
    """MSE/MAE of the final [0, 1] predictions vs targets — comparable across continuous_mode."""
    values = continuous_predictions(continuous_pred)
    return {
        "mse": F.mse_loss(values, continuous_target).item(),
        "mae": F.l1_loss(values, continuous_target).item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
def build(model_cfg: dict, frame_cfg: dict, prev_params_dim: int) -> FramePredictor:
    """
    Factory function.

    model_cfg  : config.yaml model: block
    frame_cfg  : config.yaml frame block containing params and categorical_params
    prev_params_dim : total param count from all preceding frames (computed by trainer)
    """
    categorical_params = {
        k: list(v) for k, v in frame_cfg.get("categorical_params", {}).items()
    }
    continuous_params  = [p for p in frame_cfg.get("params", []) if p not in categorical_params]

    # audio_channels: 1 = mono, 2 = stereo (P1). in_channels = audio_channels * 2
    # (mel_full + mel_prev), frame-level override wins over model-level default.
    audio_channels = int(frame_cfg.get("audio_channels", model_cfg.get("audio_channels", 1)))
    in_channels = audio_channels * 2

    model = FramePredictor(
        n_mels             = int(model_cfg.get("n_mels", 128)),
        prev_params_dim    = prev_params_dim,
        continuous_params  = continuous_params,
        categorical_params = categorical_params,
        in_channels        = in_channels,
        cnn_channels       = list(model_cfg.get("cnn_channels", [32, 64, 128])),
        cond_dim           = int(model_cfg.get("cond_dim",   64)),
        fc_hidden          = int(model_cfg.get("fc_hidden",  256)),
        dropout            = float(model_cfg.get("dropout",  0.3)),
        pool_mode          = str(model_cfg.get("pool_mode",  "stats")),
        pool_f_out         = model_cfg.get("pool_f_out"),
        temporal           = str(model_cfg.get("temporal",   "none")),
        continuous_mode    = str(model_cfg.get("continuous_mode", "regression")),
        n_bins             = int(model_cfg.get("n_bins", 32)),
    )
    model.per_param_loss_weight = bool(model_cfg.get("per_param_loss_weight", False))
    return model


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import io

    N_MELS    = 128
    BATCH     = 8
    PREV_DIM  = 12

    cont_params = ["osc_a_level", "osc_a_detune", "osc_a_fine"]
    cat_params  = {"osc_a_wave": ["sine", "saw", "square", "tri"]}

    configs = [
        dict(in_channels=2, pool_mode="freq",  temporal="none"),
        dict(in_channels=2, pool_mode="stats", temporal="none"),
        dict(in_channels=2, pool_mode="freq",  temporal="gru"),
        dict(in_channels=2, pool_mode="stats", temporal="gru"),
        dict(in_channels=4, pool_mode="stats", temporal="gru"),  # stereo (P1)
    ]

    N_BINS = 32

    for continuous_mode in ["regression", "classification"]:
        for cfg in configs:
            model = FramePredictor(
                n_mels             = N_MELS,
                prev_params_dim    = PREV_DIM,
                continuous_params  = cont_params,
                categorical_params = cat_params,
                continuous_mode    = continuous_mode,
                n_bins             = N_BINS,
                **cfg,
            )
            tag = f"mode={continuous_mode} in_ch={cfg['in_channels']} pool={cfg['pool_mode']} temporal={cfg['temporal']}"
            print(f"model011 [{tag}] | params: {model.num_parameters():,}")

            if continuous_mode == "classification":
                expected_cont_shape = (BATCH, len(cont_params), N_BINS)
            else:
                expected_cont_shape = (BATCH, len(cont_params))

            half_ch = cfg["in_channels"] // 2
            for T in [128, 256, 512]:
                mel_f = torch.randn(BATCH, half_ch, N_MELS, T)
                mel_p = torch.randn(BATCH, half_ch, N_MELS, T)
                prev  = torch.randn(BATCH, PREV_DIM)
                cont_out, cat_outs = model(mel_f, mel_p, prev)
                assert cont_out.shape == expected_cont_shape, f"cont shape: {cont_out.shape}"
                if continuous_mode == "regression":
                    assert cont_out.min() >= 0.0 and cont_out.max() <= 1.0, "cont out of [0,1]"
                for name, logits in cat_outs.items():
                    assert logits.shape == (BATCH, len(cat_params[name])), f"cat {name} shape: {logits.shape}"
                print(f"  T={T:3d} → cont {tuple(cont_out.shape)}  cat {[(k, tuple(v.shape)) for k,v in cat_outs.items()]}  ✓")

            # Zero prev_params (first frame)
            cont_out, cat_outs = model(
                torch.randn(BATCH, half_ch, N_MELS, 256),
                torch.zeros(BATCH, half_ch, N_MELS, 256),
                torch.zeros(BATCH, PREV_DIM),
            )
            print(f"  Zero prev_params → cont {tuple(cont_out.shape)}  ✓")

            # Loss smoke test
            cont_target = torch.rand(BATCH, len(cont_params))
            cat_targets = {name: torch.randint(0, len(classes), (BATCH,))
                           for name, classes in cat_params.items()}
            loss = compute_loss(
                cont_out, cat_outs, cont_target, cat_targets,
                per_param_loss_weight=model.per_param_loss_weight,
            )
            print(f"  Loss: {loss.item():.4f}  ✓")

            # Final-value metrics (comparable across continuous_mode)
            values = continuous_predictions(cont_out)
            assert values.shape == (BATCH, len(cont_params)), f"values shape: {values.shape}"
            assert values.min() >= 0.0 and values.max() <= 1.0, "values out of [0,1]"
            metrics = continuous_metrics(cont_out, cont_target)
            print(f"  Metrics: mse={metrics['mse']:.4f} mae={metrics['mae']:.4f}  ✓")

            # Per-param loss weighting smoke test (regression only)
            if continuous_mode == "regression":
                loss_weighted = compute_loss(cont_out, cat_outs, cont_target, cat_targets, per_param_loss_weight=True)
                print(f"  Loss (per_param_loss_weight): {loss_weighted.item():.4f}  ✓")

            # ONNX export check (opset 17, GRU-compatible)
            model.eval()
            dummy = (
                torch.randn(1, half_ch, N_MELS, 256),
                torch.zeros(1, half_ch, N_MELS, 256),
                torch.zeros(1, PREV_DIM),
            )
            buf = io.BytesIO()
            torch.onnx.export(
                model, dummy, buf,
                input_names=["mel_full", "mel_prev", "prev_params"],
                output_names=["continuous_out"] + list(cat_params.keys()),
                opset_version=17,
                dynamic_axes={"mel_full": {0: "batch", 3: "time"}, "mel_prev": {0: "batch", 3: "time"}},
            )
            print(f"  ONNX export ({len(buf.getvalue())} bytes)  ✓\n")

    print("All checks passed.")
