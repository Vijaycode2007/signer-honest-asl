"""Landmark sequence network used by the J/Z motion classifier."""
import torch
from torch import nn

from . import landmarks as LM


def build(vocab_size: int, in_dim: int = LM.FEATURE_DIM, hidden: int = 192,
          layers: int = 2, dropout: float = 0.3) -> nn.Module:
    """A projection, a bidirectional GRU, a per-frame head over vocab + blank."""
    class CTCNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                                      nn.GELU(), nn.Dropout(dropout))
            self.rnn = nn.GRU(hidden, hidden, layers, batch_first=True,
                              bidirectional=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden * 2, vocab_size + 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.rnn(self.proj(x))
            return self.head(h)

    return CTCNet()
