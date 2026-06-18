"""
models/model012.py

Architecture: 2D CNN + GRU — Door Classifier for sequential frame prediction pipeline.
GRU-enhanced version of model002. Receives the full render mel spectrogram and predicts:
  - 5 binary bypass flags  (Sigmoid)
  - active_lfo_count as 5-class prediction (Softmax)

Input:  (batch, n_mels, T)        — collate_variable_length output, T variable
        unsqueezed to (batch, 1, n_mels, T) inside forward()
Output: bypass_pred  (batch, n_bypass_flags)   — Sigmoid
        lfo_count_pred (batch, lfo_count_classes) — Softmax
"""

from __future__ import annotations

from typing import List, Tuple

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
class Model012(nn.Module):
    """
    2D CNN + GRU door classifier — bypass flags + LFO count head.

    Parameters
    ----------
    n_bypass_flags    : int         — number of binary bypass outputs
    lfo_count_classes : int         — number of LFO count classes (typically 5)
    cnn_channels      : List[int]   — out channels for each conv block (3 blocks)
    fc_hidden         : int         — FC hidden dim after fusion
    dropout           : float
    """

    def __init__(
            self,
            n_bypass_flags:    int,
            lfo_count_classes: int,
            cnn_channels:      List[int] = (32, 64, 128),
            fc_hidden:         int       = 256,
            dropout:           float     = 0.3,
    ) -> None:
        super().__init__()

        self.n_bypass_flags    = n_bypass_flags
        self.lfo_count_classes = lfo_count_classes

        # ── CNN stack: (B, 1, n_mels, T) → (B, C, H', W') ────────────────────
        cnn_layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            cnn_layers.append(CNNBlock2d(in_ch, out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        feat_dim = cnn_channels[-1]  # 128 by default

        # ── GRU over time, operating on freq-collapsed CNN features ─────────
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=feat_dim, batch_first=True)

        # ── Fusion: gru_out (feat_dim) + stats_pool (2 * feat_dim) ───────────
        fusion_dim = feat_dim + 2 * feat_dim

        # ── Shared FC ─────────────────────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Head A: bypass flags ───────────────────────────────────────────────
        self.bypass_head = nn.Sequential(
            nn.Linear(fc_hidden, n_bypass_flags),
            nn.Sigmoid(),
        )

        # ── Head B: LFO count ─────────────────────────────────────────────────
        # Softmax uygulanir; CrossEntropyLoss logit beklediginden
        # compute_loss logit versionu kullanir — head Softmax uygular sadece
        # inference icin
        self.lfo_head = nn.Linear(fc_hidden, lfo_count_classes)

    # ── Feature extraction ──────────────────────────────────────────────────────
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_mels, T)  — collate_variable_length output
        returns fused features (B, fc_hidden)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)          # (B, 1, n_mels, T)

        x = self.cnn(x)                 # (B, C, H', T')

        # Stats pool over (freq, time)
        stats_mean = x.mean(dim=(2, 3))             # (B, C)
        stats_std  = x.std(dim=(2, 3))              # (B, C)
        stats_pool = torch.cat([stats_mean, stats_std], dim=1)  # (B, 2C)

        # Freq collapse: mean over freq axis only → (B, C, T')
        x_freq = x.mean(dim=2)          # (B, C, T')
        x_seq  = x_freq.transpose(1, 2) # (B, T', C)

        _, h_n = self.gru(x_seq)        # h_n: (1, B, C)
        gru_out = h_n[-1]                # (B, C)

        fused = torch.cat([gru_out, stats_pool], dim=1)  # (B, 3C)
        return self.shared(fused)                        # (B, fc_hidden)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, n_mels, T)  — collate_variable_length output
        returns:
          bypass_pred   (B, n_bypass_flags)    — Sigmoid, [0,1]
          lfo_count_out (B, lfo_count_classes) — Softmax probabilities
        """
        feat = self._features(x)

        bypass_pred   = self.bypass_head(feat)                    # (B, n_bypass_flags)
        lfo_logits    = self.lfo_head(feat)                       # (B, lfo_count_classes)
        lfo_count_out = F.softmax(lfo_logits, dim=-1)             # (B, lfo_count_classes)

        return bypass_pred, lfo_count_out

    def forward_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loss hesaplama icin: bypass raw (pre-sigmoid) + lfo logits dondurur.
        BCEWithLogitsLoss ve CrossEntropyLoss numerically daha stabil.
        """
        feat = self._features(x)

        bypass_logits = self.bypass_head[0](feat)   # Linear only, no Sigmoid
        lfo_logits    = self.lfo_head(feat)

        return bypass_logits, lfo_logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(
    bypass_pred:       torch.Tensor,          # (B, n_bypass_flags)  — raw logits
    lfo_count_pred:    torch.Tensor,          # (B, lfo_count_classes) — raw logits
    bypass_target:     torch.Tensor,          # (B, n_bypass_flags)  — float 0/1
    lfo_count_target:  torch.Tensor,          # (B,) — integer class index (long)
    loss_weights:      dict = None,
    lfo_class_weights: torch.Tensor = None,   # (lfo_count_classes,) — inverse freq weights
) -> torch.Tensor:
    w_bypass = 1.0
    w_lfo    = 1.0
    if loss_weights is not None:
        w_bypass = float(loss_weights.get("bypass", 1.0))
        w_lfo    = float(loss_weights.get("lfo",    1.0))

    bce_loss = F.binary_cross_entropy_with_logits(bypass_pred, bypass_target)
    ce_loss  = F.cross_entropy(
        lfo_count_pred,
        lfo_count_target.long(),
        weight=lfo_class_weights,
    )

    return w_bypass * bce_loss + w_lfo * ce_loss


# ─────────────────────────────────────────────────────────────────────────────
def build(model_cfg: dict, door_cfg: dict) -> Model012:
    """
    Factory function.

    model_cfg: config.yaml'daki model: blogu
      cnn_channels: [32, 64, 128]
      fc_hidden:    256
      dropout:      0.3

    door_cfg: config.yaml'daki door: blogu
      bypass_columns:     [bypass_osc_b, bypass_noise, bypass_filter, bypass_reverb, bypass_delay]
      lfo_count_classes:  5

    Not: lfo_class_weights hesaplamasi burada degil, trainer'da yapilir.
    """
    n_bypass_flags    = len(door_cfg["bypass_columns"])
    lfo_count_classes = int(door_cfg["lfo_count_classes"])

    return Model012(
        n_bypass_flags    = n_bypass_flags,
        lfo_count_classes = lfo_count_classes,
        cnn_channels      = list(model_cfg.get("cnn_channels", [32, 64, 128])),
        fc_hidden         = int(model_cfg.get("fc_hidden",   256)),
        dropout           = float(model_cfg.get("dropout",   0.3)),
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _model_cfg = {"cnn_channels": [32, 64, 128], "fc_hidden": 256, "dropout": 0.3}
    _door_cfg  = {
        "bypass_columns": [
            "bypass_osc_b", "bypass_noise", "bypass_filter",
            "bypass_reverb", "bypass_delay",
        ],
        "lfo_count_classes": 5,
    }

    model = build(_model_cfg, _door_cfg)
    print(f"model012 | params: {model.num_parameters():,}")

    # Variable-T input test (collate_variable_length output shape)
    for T in [128, 256, 512]:
        x = torch.randn(8, 128, T)
        bypass_out, lfo_out = model(x)
        assert bypass_out.shape == (8, 5), f"bypass shape: {bypass_out.shape}"
        assert lfo_out.shape    == (8, 5), f"lfo shape: {lfo_out.shape}"
        assert bypass_out.min() >= 0.0 and bypass_out.max() <= 1.0, "bypass out of [0,1]"
        print(f"  Input (8, 128, {T:3d}) → bypass {tuple(bypass_out.shape)}  lfo {tuple(lfo_out.shape)}  ✓")

    # Loss smoke test
    bypass_logits, lfo_logits = model.forward_logits(torch.randn(8, 128, 256))
    loss = compute_loss(
        bypass_logits, lfo_logits,
        torch.randint(0, 2, (8, 5)).float(),
        torch.randint(0, 5, (8,)),
    )
    print(f"  Loss: {loss.item():.4f}  ✓")

    print("All checks passed.")
