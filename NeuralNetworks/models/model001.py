"""
models/model001.py

Architecture: 1D CNN + GRU hybrid
- CNN:  extracts local temporal patterns from spectrogram frames
- GRU:  captures sequential / long-range temporal structure (ADSR ordering)
- FC:   maps GRU hidden state to parameter predictions

Input:  (batch, n_features, T)   — T is variable length
Output: (batch, output_dim)      — values in [0, 1] via Sigmoid

Compatible with any output_dim (set by config).
Compatible with any input feature dimension (mel / mfcc / stft).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
class CNNBlock(nn.Module):
    """
    Single 1D CNN block:
      Conv1d → BatchNorm → ReLU → optional MaxPool
    Operates on the time axis.
    """

    def __init__(
            self,
            in_channels:  int,
            out_channels: int,
            kernel_size:  int = 3,
            pool:         bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,   # same padding — time dim korunur
            bias=False,
        )
        self.bn      = nn.BatchNorm1d(out_channels)
        self.act     = nn.ReLU(inplace=True)
        self.pool    = nn.MaxPool1d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        return self.pool(self.act(self.bn(self.conv(x))))


# ─────────────────────────────────────────────────────────────────────────────
class Model001(nn.Module):
    """
    CNN + GRU hybrid synthesizer parameter estimator.

    Parameters
    ----------
    input_dim   : int          — feature dimension (n_mels / n_mfcc / stft bins)
    output_dim  : int          — number of synth parameters to predict
    cnn_channels: List[int]    — output channels per CNN block
    gru_hidden  : int          — GRU hidden size
    gru_layers  : int          — number of stacked GRU layers
    fc_hidden   : int          — hidden size of FC head
    dropout     : float        — dropout applied after GRU and in FC
    """

    def __init__(
            self,
            input_dim:    int,
            output_dim:   int,
            cnn_channels: List[int] = (32, 64, 128),
            gru_hidden:   int       = 256,
            gru_layers:   int       = 2,
            fc_hidden:    int       = 128,
            dropout:      float     = 0.2,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        # ── CNN stack ─────────────────────────────────────────────────────────
        # Input: (B, input_dim, T)
        # Her blok time axis'i yarıya indirir (MaxPool2).
        # Son blokta pool yok — GRU uzunluğunu tamamen ezmemek için.
        cnn_layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(cnn_channels):
            pool = (i < len(cnn_channels) - 1)   # son blokta pool yok
            cnn_layers.append(CNNBlock(in_ch, out_ch, kernel_size=3, pool=pool))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # ── GRU ───────────────────────────────────────────────────────────────
        # Input: (T', B, cnn_channels[-1])  — batch_first=False
        # Output hidden: (gru_layers, B, gru_hidden)
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,              # (B, T', C) kolaylık için
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.gru_dropout = nn.Dropout(dropout)

        # ── FC head ───────────────────────────────────────────────────────────
        # Son GRU hidden state → parametreler
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, output_dim),
            nn.Sigmoid(),    # tüm parametreler [0, 1]
        )

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim, T)  — variable T
        returns: (B, output_dim)
        """
        # CNN: (B, input_dim, T) → (B, cnn_channels[-1], T')
        x = self.cnn(x)

        # GRU bekliyor: (B, T', C)
        x = x.permute(0, 2, 1)                      # (B, T', C)
        _, hidden = self.gru(x)                      # hidden: (layers, B, hidden)

        # Son katmanın hidden state'ini al
        last_hidden = hidden[-1]                     # (B, gru_hidden)
        last_hidden = self.gru_dropout(last_hidden)

        # FC head
        return self.head(last_hidden)                # (B, output_dim)

    # ── Param count ───────────────────────────────────────────────────────────
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
def build(model_cfg: dict, output_dim: int) -> Model001:
    """
    Factory function — train.py, evaluate.py, export_onnx.py buradan cagirir.

    model_cfg ornegi (config.yaml'daki model: blogu):
      input_dim:    128        # feature extraction'dan otomatik gelmeli
      cnn_channels: [32, 64, 128]
      gru_hidden:   256
      gru_layers:   2
      fc_hidden:    128
      dropout:      0.2

    input_dim config'de yoksa feature type'tan cikarilir —
    ancak en temiz yol config'e yazmak.
    """
    input_dim = model_cfg.get("input_dim", None)
    if input_dim is None:
        raise ValueError(
            "model.input_dim must be set in config.yaml. "
            "Set it to n_mels (mel_spectrogram), n_mfcc (mfcc), "
            "or 1 + n_fft // 2 (stft)."
        )

    return Model001(
        input_dim    = int(input_dim),
        output_dim   = output_dim,
        cnn_channels = list(model_cfg.get("cnn_channels", [32, 64, 128])),
        gru_hidden   = int(model_cfg.get("gru_hidden",  256)),
        gru_layers   = int(model_cfg.get("gru_layers",  2)),
        fc_hidden    = int(model_cfg.get("fc_hidden",   128)),
        dropout      = float(model_cfg.get("dropout",   0.2)),
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hizli smoke test
    model = Model001(input_dim=128, output_dim=4)
    print(f"Model001  |  params: {model.num_parameters():,}")

    # Variable-length input test
    for T in [128, 256, 512]:
        x   = torch.randn(8, 128, T)
        out = model(x)
        assert out.shape == (8, 4), f"Shape mismatch: {out.shape}"
        assert out.min() >= 0.0 and out.max() <= 1.0, "Output out of [0,1]"
        print(f"  Input (8, 128, {T:3d}) → Output {tuple(out.shape)}  ✓")

    # Farkli output_dim testi
    for odim in [2, 4, 8, 19]:
        m   = Model001(input_dim=128, output_dim=odim)
        out = m(torch.randn(4, 128, 200))
        assert out.shape == (4, odim)
        print(f"  output_dim={odim:2d}  ✓")

    print("\nAll checks passed.")