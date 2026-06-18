"""
Training utilities for the GRU SMILES generator.

Implements:
    SMILESDataset  — PyTorch Dataset that tokenises and pads SMILES sequences
    train_epoch    — one forward+backward pass over a DataLoader
    validate       — loss-only evaluation (no grad)
    train_model    — full training loop with early stopping + checkpointing
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.generation.model import GRUGenerator
from src.generation.tokenizer import SMILESTokenizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────


class SMILESDataset(Dataset):
    """
    Stores tokenised + padded SMILES sequences for language model training.

    Sequences longer than max_len (including SOS and EOS) are silently dropped.
    """

    def __init__(
        self,
        smiles_list: list[str],
        tokenizer: SMILESTokenizer,
        max_len: int = 120,
    ) -> None:
        self.pad_idx = tokenizer.pad_idx
        self.samples: list[list[int]] = []
        dropped = 0
        for smi in smiles_list:
            ids = tokenizer.encode(smi, add_sos=True, add_eos=True)
            if len(ids) <= max_len + 1:  # seq = SOS + tokens + EOS; input is [:-1]
                self.samples.append(ids)
            else:
                dropped += 1
        if dropped:
            logger.debug(
                f"SMILESDataset: dropped {dropped} sequences > max_len={max_len}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[int]:
        return self.samples[idx]

    def get_collate_fn(self):
        """Return a collate function that pads a batch and splits into x/y."""
        pad_idx = self.pad_idx

        def collate_fn(batch: list[list[int]]):
            max_len = max(len(s) for s in batch)
            padded = [s + [pad_idx] * (max_len - len(s)) for s in batch]
            t = torch.tensor(padded, dtype=torch.long)
            # x = all tokens except the last; y = all tokens except the first
            return t[:, :-1], t[:, 1:]

        return collate_fn


# ── Training loop ─────────────────────────────────────────────────────────────


def train_epoch(
    model: GRUGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(x.size(0), device)
        logits, _ = model(x, hidden)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(
    model: GRUGenerator,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            hidden = model.init_hidden(x.size(0), device)
            logits, _ = model(x, hidden)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


# ── Full training run ─────────────────────────────────────────────────────────


def train_model(
    smiles_list: list[str],
    tokenizer: SMILESTokenizer,
    save_dir: Path,
    *,
    embed_dim: int = 128,
    hidden_dim: int = 512,
    num_layers: int = 3,
    dropout: float = 0.1,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 10,
    max_len: int = 120,
    val_fraction: float = 0.10,
    seed: int = 42,
    device_str: str = "cpu",
    init_ckpt: str | Path | None = None,
    ckpt_name: str = "pretrained_gru.pt",
) -> dict[str, Any]:
    """
    Train the GRU generator and save the best checkpoint.

    Parameters
    ----------
    init_ckpt : optional path to an existing checkpoint to warm-start from
                (fine-tuning). The checkpoint's architecture must match the
                embed/hidden/layers/vocab passed here.
    ckpt_name : output filename written under ``save_dir``.

    Returns a summary dict with training history and final metrics.
    """
    device = torch.device(device_str)
    torch.manual_seed(seed)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = SMILESDataset(smiles_list, tokenizer, max_len=max_len)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    collate = dataset.get_collate_fn()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GRUGenerator(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=tokenizer.pad_idx,
    ).to(device)

    # ── Optional warm-start (fine-tuning) ─────────────────────────────────────
    if init_ckpt is not None:
        ckpt = torch.load(init_ckpt, map_location=device, weights_only=True)
        prev_cfg = ckpt.get("model_config", {})
        if prev_cfg.get("vocab_size") not in (None, tokenizer.vocab_size):
            raise ValueError(
                f"Checkpoint vocab_size={prev_cfg.get('vocab_size')} != "
                f"tokenizer.vocab_size={tokenizer.vocab_size}. "
                "Fine-tuning requires the same tokenizer used for pretraining."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Warm-started from {init_ckpt} (val_loss={ckpt.get('val_loss')})")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history: list[dict] = []
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / ckpt_name

    logger.info(
        f"Training GRU: vocab={tokenizer.vocab_size}, embed={embed_dim}, "
        f"hidden={hidden_dim}, layers={num_layers}, "
        f"n_train={n_train}, n_val={n_val}, device={device}, "
        f"warm_start={init_ckpt is not None}, ckpt={ckpt_name}"
    )
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_smiles_count": len(smiles_list),
                    "tokenizer_vocab_size": tokenizer.vocab_size,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"best={best_val_loss:.4f} (ep {best_epoch})  "
                f"elapsed={elapsed:.0f}s"
            )

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - t0
    logger.info(
        f"Training complete in {total_time:.0f}s. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}."
    )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "total_epochs": epoch,
        "total_time_s": round(total_time, 1),
        "n_train": n_train,
        "n_val": n_val,
        "checkpoint": str(ckpt_path),
        "warm_started": init_ckpt is not None,
        "history": history,
    }


def finetune_model(
    smiles_list: list[str],
    tokenizer: SMILESTokenizer,
    save_dir: Path,
    init_ckpt: str | Path,
    *,
    embed_dim: int = 128,
    hidden_dim: int = 512,
    num_layers: int = 3,
    dropout: float = 0.1,
    batch_size: int = 64,
    lr: float = 5e-4,
    epochs: int = 40,
    patience: int = 8,
    max_len: int = 120,
    val_fraction: float = 0.10,
    seed: int = 42,
    device_str: str = "cpu",
    ckpt_name: str = "egfr_finetuned_gru.pt",
) -> dict[str, Any]:
    """
    Fine-tune a pretrained GRU on a smaller, domain-specific corpus.

    Thin wrapper over ``train_model`` that warm-starts from ``init_ckpt`` and
    uses a lower default learning rate. The tokenizer MUST be the same one used
    to pretrain ``init_ckpt`` (so vocab indices line up).
    """
    return train_model(
        smiles_list=smiles_list,
        tokenizer=tokenizer,
        save_dir=save_dir,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        patience=patience,
        max_len=max_len,
        val_fraction=val_fraction,
        seed=seed,
        device_str=device_str,
        init_ckpt=init_ckpt,
        ckpt_name=ckpt_name,
    )
