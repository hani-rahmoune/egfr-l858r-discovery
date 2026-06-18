"""
Character-level SMILES tokenizer.

Splits SMILES strings into chemically meaningful tokens using a priority-ordered
regex: bracketed atoms ([C@@H], [NH3+], etc.) are matched before two-letter
elements (Br, Cl) before any single character.

Special tokens: <PAD> idx=0, <SOS> idx=1, <EOS> idx=2, <UNK> idx=3.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Priority-ordered: brackets first, then Br/Cl, then single-char catch-all.
_SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]"  # bracketed atoms
    r"|Br|Cl"  # two-letter elements outside brackets
    r"|@@|@"  # chirality specifiers
    r"|%\d{2}"  # ring closures > 9 (e.g. %10)
    r"|.)"  # any other single character
)

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL_TOKENS: list[str] = [PAD, SOS, EOS, UNK]


def tokenize(smiles: str) -> list[str]:
    """Split a SMILES string into a list of tokens."""
    return _SMILES_REGEX.findall(smiles)


class SMILESTokenizer:
    """
    Character-level SMILES tokenizer with a fixed special-token layout.

    Usage::

        tok = SMILESTokenizer().fit(smiles_list)
        ids = tok.encode("CCO")          # [1, 3, 3, 4, 2] — SOS...EOS
        smi = tok.decode(ids)            # "CCO"
        tok.save("tokenizer.json")
        tok2 = SMILESTokenizer.load("tokenizer.json")
    """

    def __init__(self) -> None:
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}
        self._built = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def fit(self, smiles_list: list[str]) -> "SMILESTokenizer":
        """Build vocabulary from a list of SMILES strings (in-place, chainable)."""
        counts: Counter = Counter()
        for smi in smiles_list:
            counts.update(tokenize(smi))

        self.token2idx = {}
        for tok in SPECIAL_TOKENS:
            self.token2idx[tok] = len(self.token2idx)
        for tok in sorted(counts.keys()):
            if tok not in self.token2idx:
                self.token2idx[tok] = len(self.token2idx)
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self._built = True
        return self

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[PAD]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[SOS]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[EOS]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[UNK]

    # ── Encode / decode ───────────────────────────────────────────────────────

    def encode(
        self, smiles: str, add_sos: bool = True, add_eos: bool = True
    ) -> list[int]:
        """Convert a SMILES string to a list of token indices."""
        tokens = tokenize(smiles)
        ids = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        if add_sos:
            ids = [self.sos_idx] + ids
        if add_eos:
            ids = ids + [self.eos_idx]
        return ids

    def decode(self, indices: list[int], strip_special: bool = True) -> str:
        """Convert a list of token indices back to a SMILES string."""
        tokens = [self.idx2token.get(i, UNK) for i in indices]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return "".join(tokens)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        Path(path).write_text(
            json.dumps(
                {"token2idx": self.token2idx, "special_tokens": SPECIAL_TOKENS},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SMILESTokenizer":
        """Load vocabulary from a JSON file saved by :meth:`save`."""
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.token2idx = data["token2idx"]
        tok.idx2token = {v: k for k, v in tok.token2idx.items()}
        tok._built = True
        return tok

    def __repr__(self) -> str:
        return f"SMILESTokenizer(vocab_size={self.vocab_size}, built={self._built})"
