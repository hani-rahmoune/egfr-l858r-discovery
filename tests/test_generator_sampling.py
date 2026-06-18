"""
Unit tests for the GRU generator model, trainer dataset, and sampler.

Uses a tiny randomly-initialised model — no training required, no checkpoint files.
PyTorch is required; tests are skipped if it is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from src.generation.model import GRUGenerator
from src.generation.sampler import evaluate_metrics, sample_smiles
from src.generation.tokenizer import SMILESTokenizer
from src.generation.trainer import SMILESDataset

# ── Shared fixtures ───────────────────────────────────────────────────────────

_CORPUS = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(=O)Nc1ccccc1",
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "C=CC(=O)Nc1cccc(Nc2ncnc3ccc(CC#N)cc23)c1",
]

_TRAIN_SMILES = {"CCO", "c1ccccc1", "CC(=O)O"}


def _make_tokenizer() -> SMILESTokenizer:
    return SMILESTokenizer().fit(_CORPUS)


def _make_model(tokenizer: SMILESTokenizer) -> GRUGenerator:
    return GRUGenerator(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        pad_idx=tokenizer.pad_idx,
    )


# ── GRUGenerator ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGRUGenerator:
    def test_forward_output_shape(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        batch, seq = 4, 10
        x = torch.randint(0, tok.vocab_size, (batch, seq))
        logits, hidden = model(x)
        assert logits.shape == (batch, seq, tok.vocab_size)

    def test_hidden_shape(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        batch, seq = 3, 8
        x = torch.randint(0, tok.vocab_size, (batch, seq))
        _, hidden = model(x)
        assert hidden.shape == (model.num_layers, batch, model.hidden_dim)

    def test_init_hidden_shape(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        device = torch.device("cpu")
        h = model.init_hidden(5, device)
        assert h.shape == (model.num_layers, 5, model.hidden_dim)
        assert h.sum().item() == 0.0

    def test_forward_with_hidden(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        x = torch.randint(0, tok.vocab_size, (2, 5))
        h = model.init_hidden(2, torch.device("cpu"))
        logits, h2 = model(x, h)
        assert logits.shape == (2, 5, tok.vocab_size)
        assert h2.shape == h.shape

    def test_config_dict_keys(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        cfg = model.config()
        assert set(cfg.keys()) >= {
            "vocab_size",
            "embed_dim",
            "hidden_dim",
            "num_layers",
        }

    def test_no_nan_in_logits(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        x = torch.randint(0, tok.vocab_size, (2, 5))
        logits, _ = model(x)
        assert not torch.isnan(logits).any()

    def test_forward_batch_size_one(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        x = torch.randint(0, tok.vocab_size, (1, 3))
        logits, _ = model(x)
        assert logits.shape[0] == 1

    def test_different_batch_sizes(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        for bs in (1, 4, 16):
            x = torch.randint(0, tok.vocab_size, (bs, 5))
            logits, _ = model(x)
            assert logits.shape[0] == bs


# ── SMILESDataset ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSMILESDataset:
    def test_len_positive(self):
        tok = _make_tokenizer()
        ds = SMILESDataset(_CORPUS, tok)
        assert len(ds) > 0

    def test_getitem_is_list_of_ints(self):
        tok = _make_tokenizer()
        ds = SMILESDataset(_CORPUS, tok)
        item = ds[0]
        assert isinstance(item, list)
        assert all(isinstance(i, int) for i in item)

    def test_starts_with_sos(self):
        tok = _make_tokenizer()
        ds = SMILESDataset(_CORPUS, tok)
        item = ds[0]
        assert item[0] == tok.sos_idx

    def test_ends_with_eos(self):
        tok = _make_tokenizer()
        ds = SMILESDataset(_CORPUS, tok)
        item = ds[0]
        assert item[-1] == tok.eos_idx

    def test_max_len_filters_long(self):
        tok = _make_tokenizer()
        long_smi = ["C" * 200]
        ds = SMILESDataset(long_smi, tok, max_len=10)
        assert len(ds) == 0

    def test_collate_fn_produces_tensors(self):
        tok = _make_tokenizer()
        ds = SMILESDataset(_CORPUS, tok)
        collate = ds.get_collate_fn()
        batch = [ds[i] for i in range(min(3, len(ds)))]
        x, y = collate(batch)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_collate_fn_x_y_offset(self):
        tok = _make_tokenizer()
        ds = SMILESDataset(["CCO"], tok)
        collate = ds.get_collate_fn()
        batch = [ds[0]]
        x, y = collate(batch)
        # x is all tokens except last; y is all tokens except first
        # For single seq both have same length (seq_len - 1)
        assert x.shape[1] == y.shape[1]

    def test_collate_fn_padding(self):
        tok = _make_tokenizer()
        # Use SMILES of very different lengths to force padding
        ds = SMILESDataset(
            ["CCO", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"], tok
        )
        collate = ds.get_collate_fn()
        batch = [ds[0], ds[1]]
        x, y = collate(batch)
        # Both rows must have the same length (padded to longest)
        assert x.shape[0] == 2
        assert x.shape[1] == y.shape[1]


# ── sample_smiles ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSampleSmiles:
    def test_returns_n_strings(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        results = sample_smiles(model, tok, n=5, max_len=20)
        assert len(results) == 5

    def test_returns_strings(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        results = sample_smiles(model, tok, n=3, max_len=15)
        assert all(isinstance(s, str) for s in results)

    def test_temperature_one(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        results = sample_smiles(model, tok, n=2, max_len=10, temperature=1.0)
        assert len(results) == 2

    def test_temperature_low(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        results = sample_smiles(model, tok, n=2, max_len=10, temperature=0.5)
        assert len(results) == 2

    def test_max_len_respected(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        max_len = 5
        results = sample_smiles(model, tok, n=10, max_len=max_len)
        # Each result must be at most max_len tokens when tokenised
        from src.generation.tokenizer import tokenize

        for smi in results:
            assert len(tokenize(smi)) <= max_len

    def test_n_zero(self):
        tok = _make_tokenizer()
        model = _make_model(tok)
        assert sample_smiles(model, tok, n=0) == []


# ── evaluate_metrics ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestEvaluateMetrics:
    def test_required_keys(self):
        m = evaluate_metrics(["CCO", "not_smiles"])
        for key in ("n_generated", "n_valid", "n_unique", "validity", "uniqueness"):
            assert key in m

    def test_validity_known_smiles(self):
        m = evaluate_metrics(["CCO", "c1ccccc1", "CC"])
        assert m["n_valid"] == 3
        assert m["validity"] == pytest.approx(1.0)

    def test_validity_invalid_smiles(self):
        m = evaluate_metrics(["ZZZZZ", "!!!"])
        assert m["n_valid"] == 0
        assert m["validity"] == 0.0

    def test_uniqueness_all_same(self):
        m = evaluate_metrics(["CCO", "CCO", "CCO"])
        assert m["n_unique"] == 1
        assert m["uniqueness"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_uniqueness_all_different(self):
        m = evaluate_metrics(["CCO", "c1ccccc1", "CCC"])
        assert m["uniqueness"] == pytest.approx(1.0)

    def test_novelty_with_train_set(self):
        train = {"CCO", "c1ccccc1"}
        m = evaluate_metrics(["CCO", "CCC", "CCCO"], train_smiles=train)
        # CCO is in train, CCC and CCCO are novel
        assert m["n_novel"] is not None
        assert m["n_novel"] >= 1

    def test_novelty_none_without_train_set(self):
        m = evaluate_metrics(["CCO"])
        assert m["novelty"] is None
        assert m["n_novel"] is None

    def test_empty_input(self):
        m = evaluate_metrics([])
        assert m["n_generated"] == 0
        assert m["validity"] == 0.0

    def test_mixed_valid_invalid(self):
        m = evaluate_metrics(["CCO", "INVALID", "c1ccccc1"])
        assert m["n_valid"] == 2
        assert m["n_generated"] == 3
        assert m["validity"] == pytest.approx(2 / 3, abs=0.01)

    def test_canonical_deduplication(self):
        # RDKit canonicalises both to same SMILES
        m = evaluate_metrics(["OCC", "CCO"])
        # Both are ethanol — should appear as 1 unique
        assert m["n_unique"] == 1
