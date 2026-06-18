"""
Unit tests for src/generation/tokenizer.py.

All tests are pure-Python (no PyTorch dependency).
Covers: tokenize(), SMILESTokenizer.fit/encode/decode/save/load, edge cases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.tokenizer import (
    EOS,
    PAD,
    SOS,
    SPECIAL_TOKENS,
    SMILESTokenizer,
    tokenize,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
OSIMERTINIB = "COc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1NC(C)=O"
BROMOBENZENE = "Brc1ccccc1"  # Br must be a single token
CHLOROBENZENE = "Clc1ccccc1"  # Cl must be a single token
CHIRAL = "OC[C@@H](O)CO"  # chirality in bracket
RING_CLOSURE = "C1CCCCC1"  # single-digit ring closure
ETHANOL = "CCO"
CORPUS = [ASPIRIN, GEFITINIB, OSIMERTINIB, BROMOBENZENE, CHLOROBENZENE]


# ── tokenize() ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTokenize:
    def test_returns_list(self):
        assert isinstance(tokenize(ETHANOL), list)

    def test_simple_smiles(self):
        # CCO -> ['C', 'C', 'O']
        assert tokenize("CCO") == ["C", "C", "O"]

    def test_bracket_atom_is_single_token(self):
        tokens = tokenize("[NH3+]")
        assert "[NH3+]" in tokens

    def test_bracket_chiral_is_single_token(self):
        tokens = tokenize("[C@@H]")
        assert "[C@@H]" in tokens

    def test_br_is_single_token(self):
        tokens = tokenize(BROMOBENZENE)
        assert "Br" in tokens
        assert "B" not in tokens or tokens.index("Br") != -1  # Br not split

    def test_cl_is_single_token(self):
        tokens = tokenize(CHLOROBENZENE)
        assert "Cl" in tokens

    def test_roundtrip_join(self):
        # Joining tokens back must give the original SMILES
        for smi in [ASPIRIN, GEFITINIB, BROMOBENZENE, RING_CLOSURE]:
            assert "".join(tokenize(smi)) == smi

    def test_empty_string(self):
        assert tokenize("") == []

    def test_ring_closure_digits(self):
        tokens = tokenize(RING_CLOSURE)
        assert "1" in tokens

    def test_aromatic_atoms(self):
        tokens = tokenize("c1ccccc1")
        assert all(t == "c" for t in tokens if t.isalpha())

    def test_double_bond(self):
        tokens = tokenize("C=C")
        assert "=" in tokens


# ── SMILESTokenizer ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTokenizerFit:
    def test_vocab_size_positive(self):
        tok = SMILESTokenizer().fit(CORPUS)
        assert tok.vocab_size > 0

    def test_special_tokens_present(self):
        tok = SMILESTokenizer().fit(CORPUS)
        for sp in SPECIAL_TOKENS:
            assert sp in tok.token2idx

    def test_pad_idx_zero(self):
        tok = SMILESTokenizer().fit(CORPUS)
        assert tok.pad_idx == 0

    def test_sos_idx_one(self):
        tok = SMILESTokenizer().fit(CORPUS)
        assert tok.sos_idx == 1

    def test_eos_idx_two(self):
        tok = SMILESTokenizer().fit(CORPUS)
        assert tok.eos_idx == 2

    def test_vocab_contains_common_tokens(self):
        tok = SMILESTokenizer().fit(CORPUS)
        for ch in ["C", "c", "N", "O", "(", ")"]:
            assert ch in tok.token2idx

    def test_fit_returns_self(self):
        tok = SMILESTokenizer()
        result = tok.fit(CORPUS)
        assert result is tok

    def test_repr(self):
        tok = SMILESTokenizer().fit(CORPUS)
        assert "vocab_size" in repr(tok)


@pytest.mark.unit
class TestTokenizerEncode:
    def setup_method(self):
        self.tok = SMILESTokenizer().fit(CORPUS)

    def test_encode_returns_list_of_ints(self):
        ids = self.tok.encode(ETHANOL)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_starts_with_sos(self):
        ids = self.tok.encode(ETHANOL, add_sos=True)
        assert ids[0] == self.tok.sos_idx

    def test_encode_ends_with_eos(self):
        ids = self.tok.encode(ETHANOL, add_eos=True)
        assert ids[-1] == self.tok.eos_idx

    def test_encode_no_sos_no_eos(self):
        ids = self.tok.encode(ETHANOL, add_sos=False, add_eos=False)
        assert ids[0] != self.tok.sos_idx
        assert ids[-1] != self.tok.eos_idx

    def test_encode_length(self):
        # len = sos + n_tokens + eos
        ids = self.tok.encode(ETHANOL)
        assert len(ids) == len(tokenize(ETHANOL)) + 2

    def test_unknown_token_maps_to_unk(self):
        ids = self.tok.encode("ZZZZZ", add_sos=False, add_eos=False)
        for i in ids:
            assert i == self.tok.unk_idx


@pytest.mark.unit
class TestTokenizerDecode:
    def setup_method(self):
        self.tok = SMILESTokenizer().fit(CORPUS)

    def test_decode_roundtrip_ethanol(self):
        ids = self.tok.encode(ETHANOL)
        assert self.tok.decode(ids) == ETHANOL

    def test_decode_roundtrip_aspirin(self):
        ids = self.tok.encode(ASPIRIN)
        assert self.tok.decode(ids) == ASPIRIN

    def test_decode_roundtrip_bromobenzene(self):
        ids = self.tok.encode(BROMOBENZENE)
        assert self.tok.decode(ids) == BROMOBENZENE

    def test_decode_strips_special_by_default(self):
        ids = [self.tok.sos_idx, self.tok.token2idx.get("C", 5), self.tok.eos_idx]
        result = self.tok.decode(ids, strip_special=True)
        assert SOS not in result
        assert EOS not in result

    def test_decode_keeps_special_when_not_stripped(self):
        ids = [self.tok.sos_idx, self.tok.eos_idx]
        result = self.tok.decode(ids, strip_special=False)
        assert SOS in result

    def test_decode_empty_ids(self):
        assert self.tok.decode([]) == ""


@pytest.mark.unit
class TestTokenizerPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        tok = SMILESTokenizer().fit(CORPUS)
        p = tmp_path / "tokenizer.json"
        tok.save(p)
        tok2 = SMILESTokenizer.load(p)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.pad_idx == tok.pad_idx
        assert tok2.sos_idx == tok.sos_idx

    def test_saved_file_is_valid_json(self, tmp_path):
        tok = SMILESTokenizer().fit(CORPUS)
        p = tmp_path / "tok.json"
        tok.save(p)
        data = json.loads(p.read_text())
        assert "token2idx" in data

    def test_encode_decode_after_load(self, tmp_path):
        tok = SMILESTokenizer().fit(CORPUS)
        p = tmp_path / "t.json"
        tok.save(p)
        tok2 = SMILESTokenizer.load(p)
        ids = tok2.encode(ASPIRIN)
        assert tok2.decode(ids) == ASPIRIN

    def test_load_preserves_idx2token(self, tmp_path):
        tok = SMILESTokenizer().fit(CORPUS)
        p = tmp_path / "t.json"
        tok.save(p)
        tok2 = SMILESTokenizer.load(p)
        # idx2token must map pad_idx back to PAD
        assert tok2.idx2token[tok2.pad_idx] == PAD
