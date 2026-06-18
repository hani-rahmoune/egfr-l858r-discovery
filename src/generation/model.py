"""
GRU-based character-level SMILES generator.

Architecture: Embedding → GRU (multi-layer) → Dropout → Linear → vocab logits.
Trained with teacher forcing and cross-entropy (PAD tokens ignored via ignore_index).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUGenerator(nn.Module):
    """
    Character-level SMILES language model using a stacked GRU.

    Parameters
    ----------
    vocab_size : vocabulary size (from SMILESTokenizer.vocab_size)
    embed_dim  : embedding dimension (default 128)
    hidden_dim : GRU hidden dimension (default 512)
    num_layers : number of stacked GRU layers (default 3)
    dropout    : dropout probability applied after embedding and before output linear
                 (also used between GRU layers when num_layers > 1)
    pad_idx    : embedding padding index; those embeddings are zeroed and not updated
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x      : (batch, seq_len) token indices
        hidden : (num_layers, batch, hidden_dim) or None

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        hidden : (num_layers, batch, hidden_dim)
        """
        emb = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
        out, hidden = self.gru(emb, hidden)  # (batch, seq_len, hidden_dim)
        logits = self.fc(self.dropout(out))  # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero-initialised hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def config(self) -> dict:
        """Return constructor kwargs for serialisation."""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout.p,
        }
