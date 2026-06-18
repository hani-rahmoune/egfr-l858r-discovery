"""
Sampling and evaluation utilities for the GRU SMILES generator.

Public API
----------
load_checkpoint(ckpt_path, tokenizer_path)  -> (GRUGenerator, SMILESTokenizer)
sample_smiles(model, tokenizer, n, ...)     -> list[str]
evaluate_metrics(generated, train_smiles)   -> dict with validity/uniqueness/novelty
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.generation.model import GRUGenerator
from src.generation.tokenizer import SMILESTokenizer

# ── Checkpoint I/O ────────────────────────────────────────────────────────────


def load_checkpoint(
    ckpt_path: str | Path,
    tokenizer_path: str | Path,
    device_str: str = "cpu",
) -> tuple[GRUGenerator, SMILESTokenizer]:
    """
    Load a saved checkpoint and its corresponding tokenizer.

    Parameters
    ----------
    ckpt_path       : path to ``pretrained_gru.pt``
    tokenizer_path  : path to ``tokenizer.json``
    device_str      : torch device string (default 'cpu')

    Returns
    -------
    (model, tokenizer)
    """
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["model_config"]

    tokenizer = SMILESTokenizer.load(tokenizer_path)
    model = GRUGenerator(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg.get("embed_dim", 128),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.1),
        pad_idx=tokenizer.pad_idx,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


# ── Sampling ──────────────────────────────────────────────────────────────────


def sample_smiles(
    model: GRUGenerator,
    tokenizer: SMILESTokenizer,
    n: int,
    max_len: int = 120,
    temperature: float = 1.0,
    device_str: str = "cpu",
) -> list[str]:
    """
    Generate ``n`` SMILES strings by autoregressive sampling.

    Parameters
    ----------
    model       : trained GRUGenerator (in eval mode)
    tokenizer   : fitted SMILESTokenizer
    n           : number of molecules to generate
    max_len     : maximum number of tokens per molecule
    temperature : sampling temperature (< 1 = more conservative, > 1 = more random)
    device_str  : torch device string

    Returns
    -------
    List of raw SMILES strings (may include invalid SMILES).
    """
    device = torch.device(device_str)
    model.eval()
    results: list[str] = []

    with torch.no_grad():
        for _ in range(n):
            tokens: list[int] = [tokenizer.sos_idx]
            hidden = model.init_hidden(1, device)

            for _ in range(max_len):
                x = torch.tensor([[tokens[-1]]], device=device)
                logits, hidden = model(x, hidden)
                scaled = logits[0, 0] / temperature
                probs = torch.softmax(scaled, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())
                if next_tok == tokenizer.eos_idx:
                    break
                tokens.append(next_tok)

            results.append(tokenizer.decode(tokens, strip_special=True))

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────


def scaffold_stats(
    canonical_smiles: list[str],
    train_smiles: set[str] | None = None,
) -> dict[str, Any]:
    """
    Bemis-Murcko scaffold diversity for a list of (already canonical) SMILES.

    Returns
    -------
    dict with keys:
        n_scaffolds         : number of distinct Bemis-Murcko scaffolds
        scaffold_diversity  : n_scaffolds / n_molecules (1.0 = every molecule a
                              different scaffold; low = many copies of few cores)
        n_novel_scaffolds   : scaffolds absent from the training set (or None)
        scaffold_novelty    : n_novel_scaffolds / n_scaffolds (or None)
    """
    from src.splitting.scaffold_split import get_bemis_murcko_scaffold

    scaffolds: set[str] = set()
    for smi in canonical_smiles:
        scaf = get_bemis_murcko_scaffold(smi)
        if scaf is not None:
            scaffolds.add(scaf)

    n_mol = len(canonical_smiles)
    n_scaf = len(scaffolds)
    diversity = round(n_scaf / n_mol, 4) if n_mol else 0.0

    n_novel_scaf: int | None = None
    scaf_novelty: float | None = None
    if train_smiles is not None:
        train_scaffolds: set[str] = set()
        for smi in train_smiles:
            scaf = get_bemis_murcko_scaffold(smi)
            if scaf is not None:
                train_scaffolds.add(scaf)
        novel = scaffolds - train_scaffolds
        n_novel_scaf = len(novel)
        scaf_novelty = round(n_novel_scaf / n_scaf, 4) if n_scaf else 0.0

    return {
        "n_scaffolds": n_scaf,
        "scaffold_diversity": diversity,
        "n_novel_scaffolds": n_novel_scaf,
        "scaffold_novelty": scaf_novelty,
    }


def evaluate_metrics(
    generated: list[str],
    train_smiles: set[str] | None = None,
    compute_scaffolds: bool = True,
) -> dict[str, Any]:
    """
    Compute validity, uniqueness, novelty, and scaffold diversity for generated SMILES.

    Parameters
    ----------
    generated         : raw SMILES strings from sample_smiles()
    train_smiles      : set of canonical SMILES in the training corpus; if None,
                        novelty / scaffold_novelty are reported as None
    compute_scaffolds : if True (default), also report Bemis-Murcko scaffold stats
                        over the unique valid molecules

    Returns
    -------
    dict with keys:
        n_generated, n_valid, n_unique, n_novel,
        validity, uniqueness, novelty (float or None),
        and (when compute_scaffolds) n_scaffolds, scaffold_diversity,
        n_novel_scaffolds, scaffold_novelty
    """
    from rdkit import Chem

    valid_canonical: list[str] = []
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_canonical.append(Chem.MolToSmiles(mol))

    n_gen = len(generated)
    n_valid = len(valid_canonical)
    unique = set(valid_canonical)
    n_unique = len(unique)

    validity = round(n_valid / n_gen, 4) if n_gen else 0.0
    uniqueness = round(n_unique / n_valid, 4) if n_valid else 0.0

    n_novel: int | None = None
    novelty: float | None = None
    if train_smiles is not None:
        novel = {s for s in unique if s not in train_smiles}
        n_novel = len(novel)
        novelty = round(n_novel / n_unique, 4) if n_unique else 0.0

    result = {
        "n_generated": n_gen,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_novel": n_novel,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
    }

    if compute_scaffolds:
        result.update(scaffold_stats(sorted(unique), train_smiles=train_smiles))

    return result
