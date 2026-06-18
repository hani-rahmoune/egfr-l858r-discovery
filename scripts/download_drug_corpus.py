"""
Download and filter a large, generic drug-like SMILES corpus for generator pretraining.

Source: MOSES dataset_v1.csv (molecularsets/moses) — ~1.76M ZINC-derived,
drug-like molecules already curated for clean SMILES grammar. We stream-download
it once (cached), apply a drug-like MW/LogP filter, canonicalise + deduplicate,
and subsample to a target size feasible for CPU training.

The point of this corpus is to teach the GRU valid SMILES *grammar and diversity*
before fine-tuning on the ~1347 EGFR/ErbB2 actives. The tiny kinase corpus alone
only reaches ~56% validity; a broad base lifts it well past 90%.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/download_drug_corpus.py
    PYTHONPATH=. .venv/Scripts/python.exe scripts/download_drug_corpus.py --target 150000

Outputs:
    data/external/moses_dataset_v1.csv   — cached raw download (~84 MB)
    data/interim/drug_like_corpus.smi    — filtered canonical SMILES, one per line
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

_MOSES_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
_RAW_PATH = PROJECT_ROOT / "data" / "external" / "moses_dataset_v1.csv"
_OUT_PATH = PROJECT_ROOT / "data" / "interim" / "drug_like_corpus.smi"

# Drug-like property windows (inclusive). Deliberately broad — MOSES is already
# pre-filtered, this is a sanity gate, not an aggressive cut.
_MW_MIN, _MW_MAX = 150.0, 500.0
_LOGP_MIN, _LOGP_MAX = -1.0, 5.0


def download_raw(url: str = _MOSES_URL, dest: Path = _RAW_PATH) -> Path:
    """Stream-download the raw corpus to disk (cached; skips if present)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        logger.info(f"Using cached corpus: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    logger.info(f"Downloading {url} -> {dest} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        downloaded = 0
        while True:
            chunk = r.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (20 << 20) < (1 << 20):
                logger.info(f"  {downloaded/1e6:.0f} MB ...")
    tmp.replace(dest)
    logger.info(f"Downloaded {dest.stat().st_size/1e6:.1f} MB")
    return dest


def read_smiles_column(raw_path: Path) -> list[str]:
    """Read the SMILES column from the MOSES CSV (header: SMILES,SPLIT)."""
    smiles: list[str] = []
    with open(raw_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            smi_idx = header.index("SMILES")
        except ValueError:
            smi_idx = 0  # fall back to first column
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) > smi_idx and parts[smi_idx]:
                smiles.append(parts[smi_idx])
    logger.info(f"Read {len(smiles)} raw SMILES from {raw_path.name}")
    return smiles


def is_drug_like(mol) -> bool:
    """Apply broad MW/LogP drug-likeness gate."""
    from rdkit.Chem import Descriptors

    mw = Descriptors.MolWt(mol)
    if not (_MW_MIN <= mw <= _MW_MAX):
        return False
    logp = Descriptors.MolLogP(mol)
    if not (_LOGP_MIN <= logp <= _LOGP_MAX):
        return False
    return True


def filter_and_subsample(
    smiles: list[str],
    target: int,
    seed: int = 42,
    presample_factor: float = 1.6,
) -> list[str]:
    """
    Canonicalise, drug-like filter, dedup, and subsample to `target` molecules.

    To avoid running RDKit on all ~1.76M molecules, we first shuffle and take a
    pre-sample of ~target*presample_factor candidates, then filter that pool.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    rng = random.Random(seed)
    rng.shuffle(smiles)

    pool_size = min(len(smiles), int(target * presample_factor) + 1000)
    pool = smiles[:pool_size]
    logger.info(
        f"Filtering a pre-sampled pool of {len(pool)} candidates (target={target}) ..."
    )

    seen: set[str] = set()
    kept: list[str] = []
    for i, smi in enumerate(pool):
        if len(kept) >= target:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not is_drug_like(mol):
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        kept.append(can)
        if (i + 1) % 50000 == 0:
            logger.info(f"  scanned {i+1}/{len(pool)}, kept {len(kept)}")

    # If the pre-sample was too small to reach target, widen the scan.
    if len(kept) < target and pool_size < len(smiles):
        logger.info(f"Pre-sample yielded {len(kept)} < target; scanning remainder ...")
        for smi in smiles[pool_size:]:
            if len(kept) >= target:
                break
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not is_drug_like(mol):
                continue
            can = Chem.MolToSmiles(mol)
            if can in seen:
                continue
            seen.add(can)
            kept.append(can)

    logger.info(f"Kept {len(kept)} drug-like canonical SMILES")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + filter drug-like corpus")
    parser.add_argument(
        "--target", type=int, default=150000, help="Number of drug-like SMILES to keep"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--url", type=str, default=_MOSES_URL)
    parser.add_argument("--out", type=Path, default=_OUT_PATH)
    args = parser.parse_args()

    logger.info("=== download_drug_corpus ===")
    raw = download_raw(args.url)
    smiles = read_smiles_column(raw)
    kept = filter_and_subsample(smiles, target=args.target, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(kept)} SMILES to {args.out}")
    print(f"\nDrug-like corpus: {len(kept)} SMILES -> {args.out}")


if __name__ == "__main__":
    main()
