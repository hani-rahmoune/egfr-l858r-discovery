"""
Unit tests for warm-start / fine-tuning support in src/generation/trainer.py.

Trains a tiny base model for 1 epoch, then verifies:
  - train_model writes the checkpoint under the requested ckpt_name
  - train_model(init_ckpt=...) warm-starts without error and reuses the weights
  - finetune_model produces a distinct fine-tuned checkpoint
  - a vocab mismatch between checkpoint and tokenizer raises ValueError

PyTorch required; skipped if absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from src.generation.tokenizer import SMILESTokenizer
from src.generation.trainer import finetune_model, train_model

_BASE_CORPUS = [
    "CCO",
    "CCN",
    "CCC",
    "c1ccccc1",
    "Cc1ccccc1",
    "CC(=O)O",
    "CCOC",
    "CCCl",
    "CCBr",
    "c1ccncc1",
    "CCCC",
    "CCCCO",
]
_EGFR_CORPUS = ["COc1ccccc1", "Nc1ccccc1", "Cc1ccncc1", "CCc1ccccc1"]

_ARCH = dict(embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)


def _tok() -> SMILESTokenizer:
    # union vocab so the fine-tune corpus has no unknown tokens
    return SMILESTokenizer().fit(_BASE_CORPUS + _EGFR_CORPUS)


def _train_base(tmp_path: Path, tok: SMILESTokenizer) -> Path:
    train_model(
        smiles_list=_BASE_CORPUS,
        tokenizer=tok,
        save_dir=tmp_path,
        batch_size=4,
        lr=1e-3,
        epochs=1,
        patience=5,
        val_fraction=0.25,
        seed=0,
        ckpt_name="base.pt",
        **_ARCH,
    )
    return tmp_path / "base.pt"


@pytest.mark.unit
class TestCheckpointNaming:
    def test_train_model_writes_named_ckpt(self, tmp_path):
        tok = _tok()
        base = _train_base(tmp_path, tok)
        assert base.exists()

    def test_summary_reports_not_warm_started(self, tmp_path):
        tok = _tok()
        summary = train_model(
            smiles_list=_BASE_CORPUS,
            tokenizer=tok,
            save_dir=tmp_path,
            batch_size=4,
            lr=1e-3,
            epochs=1,
            patience=5,
            val_fraction=0.25,
            seed=0,
            ckpt_name="b.pt",
            **_ARCH,
        )
        assert summary["warm_started"] is False


@pytest.mark.unit
class TestWarmStart:
    def test_warm_start_runs(self, tmp_path):
        tok = _tok()
        base = _train_base(tmp_path, tok)
        summary = train_model(
            smiles_list=_EGFR_CORPUS,
            tokenizer=tok,
            save_dir=tmp_path,
            batch_size=4,
            lr=5e-4,
            epochs=1,
            patience=5,
            val_fraction=0.25,
            seed=0,
            init_ckpt=base,
            ckpt_name="ft.pt",
            **_ARCH,
        )
        assert summary["warm_started"] is True
        assert (tmp_path / "ft.pt").exists()

    def test_warm_start_loads_weights(self, tmp_path):
        # The fine-tuned model's first embedding row should be close to the base's
        # after a single low-LR epoch (warm-start, not random re-init).
        from src.generation.sampler import load_checkpoint

        tok = _tok()
        base = _train_base(tmp_path, tok)
        tok.save(tmp_path / "tokenizer.json")
        base_model, _ = load_checkpoint(base, tmp_path / "tokenizer.json")
        base_emb = base_model.embedding.weight.detach().clone()

        finetune_model(
            smiles_list=_EGFR_CORPUS,
            tokenizer=tok,
            save_dir=tmp_path,
            init_ckpt=base,
            batch_size=4,
            lr=1e-5,
            epochs=1,
            patience=5,
            val_fraction=0.25,
            seed=0,
            ckpt_name="ft.pt",
            **_ARCH,
        )
        ft_model, _ = load_checkpoint(tmp_path / "ft.pt", tmp_path / "tokenizer.json")
        ft_emb = ft_model.embedding.weight.detach()
        # With lr=1e-5 for one epoch, weights barely move from the warm start
        assert torch.allclose(base_emb, ft_emb, atol=1e-2)


@pytest.mark.unit
class TestFinetuneModel:
    def test_finetune_creates_checkpoint(self, tmp_path):
        tok = _tok()
        base = _train_base(tmp_path, tok)
        finetune_model(
            smiles_list=_EGFR_CORPUS,
            tokenizer=tok,
            save_dir=tmp_path,
            init_ckpt=base,
            batch_size=4,
            lr=5e-4,
            epochs=1,
            patience=5,
            val_fraction=0.25,
            seed=0,
            ckpt_name="egfr_ft.pt",
            **_ARCH,
        )
        assert (tmp_path / "egfr_ft.pt").exists()

    def test_vocab_mismatch_raises(self, tmp_path):
        tok = _tok()
        base = _train_base(tmp_path, tok)
        # A different tokenizer with a different vocab size must be rejected
        bigger = SMILESTokenizer().fit(
            _BASE_CORPUS + _EGFR_CORPUS + ["[Si]", "[Se]", "P", "I"]
        )
        if bigger.vocab_size == tok.vocab_size:
            pytest.skip("vocab sizes coincidentally equal")
        with pytest.raises(ValueError):
            train_model(
                smiles_list=_EGFR_CORPUS,
                tokenizer=bigger,
                save_dir=tmp_path,
                batch_size=4,
                lr=5e-4,
                epochs=1,
                patience=5,
                val_fraction=0.25,
                seed=0,
                init_ckpt=base,
                ckpt_name="bad.pt",
                **_ARCH,
            )
