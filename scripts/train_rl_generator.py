"""
Phase 22: REINVENT-style RL fine-tuning of the EGFR GRU generator.

Warm-starts from egfr_finetuned_gru.pt, trains for a fixed step budget using
a multi-objective reward, then reports a decisive pre-vs-post comparison table.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/train_rl_generator.py
    PYTHONPATH=. .venv/Scripts/python.exe scripts/train_rl_generator.py --n_steps 50 --batch_size 32

Prerequisites:
    scripts/finetune_generator.py   (models/generator/egfr_finetuned_gru.pt)
    scripts/train_models.py         (models/qsar/general/)
    data/interim/egfr_cleaned.csv, erbb2_cleaned.csv

Output files:
    models/generator/rl_finetuned_gru.pt      (RL-tuned agent checkpoint)
    models/generator/rl_results.json           (metrics + step log)
    logs/rl_training.log                       (CSV step log — no grep pipe)

All generated molecules are EXPLORATORY. Backbone pIC50 is in-sample;
docking should be applied post-hoc to the best post-RL hits.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from rdkit import Chem

from src.generation.reward import MoleculeReward
from src.generation.rl_trainer import (
    REINVENTTrainer,
    ScaffoldMemory,
    compare_pre_post,
    evaluate_generator,
)
from src.generation.sampler import load_checkpoint
from src.models.qsar import QSARTrainer
from src.scoring.applicability_domain import ApplicabilityDomain
from src.utils.config import load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GEN_DIR = PROJECT_ROOT / "models" / "generator"
_MODEL_DIR = PROJECT_ROOT / "models" / "qsar" / "general"
_EGFR_CSV = PROJECT_ROOT / "data" / "interim" / "egfr_cleaned.csv"
_ERBB2_CSV = PROJECT_ROOT / "data" / "interim" / "erbb2_cleaned.csv"
_LOGS_DIR = PROJECT_ROOT / "logs"
_LOG_CSV = _LOGS_DIR / "rl_training.log"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_train_smiles() -> set[str]:
    frames = []
    for csv_path in (_EGFR_CSV, _ERBB2_CSV):
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path, usecols=["canonical_smiles"]))
    if not frames:
        return set()
    raw = pd.concat(frames)["canonical_smiles"].dropna().tolist()
    result: set[str] = set()
    for smi in raw:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            result.add(Chem.MolToSmiles(mol))
    return result


def _print_metrics(label: str, m: dict) -> None:
    print(f"\n  [{label}]")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"    {k:<24}: {v:.4f}")
        elif v is not None:
            print(f"    {k:<24}: {v}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="REINVENT RL fine-tuning")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="Number of RL training steps (overrides config)",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--eval_n", type=int, default=None, help="Molecules to sample for pre/post eval"
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = load_model_config()
    rl_cfg = cfg.get("generator", {}).get("rl", {})
    n_steps = args.n_steps or rl_cfg.get("n_steps", 100)
    batch_sz = args.batch_size or rl_cfg.get("batch_size", 64)
    sigma = args.sigma or rl_cfg.get("sigma", 0.5)
    lr = args.lr or rl_cfg.get("lr", 1e-4)
    max_len = rl_cfg.get("max_len", 120)
    temp = rl_cfg.get("temperature", 1.0)
    eval_n = args.eval_n or rl_cfg.get("eval_n_sample", 512)
    eval_temp = cfg.get("generator", {}).get("sample_temperature", 0.8)
    reward_cfg = rl_cfg.get("reward", {})
    seed = rl_cfg.get("seed", 42)
    df_cfg = rl_cfg.get("diversity_filter", {})

    device = torch.device(args.device)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("Phase 22 — REINVENT RL fine-tuning")
    logger.info(f"  n_steps={n_steps}  batch_size={batch_sz}  sigma={sigma}  lr={lr}")
    logger.info(f"  eval_n={eval_n}    eval_temp={eval_temp}")
    logger.info(
        f"  diversity_filter={df_cfg.get('enabled', False)} "
        f"(bucket={df_cfg.get('bucket_size', 25)}, "
        f"min_score={df_cfg.get('min_score', 0.30)})"
    )
    logger.info("=" * 60)

    # ── Load agent + frozen prior ─────────────────────────────────────────────
    ckpt_path = _GEN_DIR / "egfr_finetuned_gru.pt"
    tok_path = _GEN_DIR / "tokenizer.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {ckpt_path}. "
            "Run scripts/finetune_generator.py first."
        )

    logger.info(f"Loading agent from {ckpt_path}")
    agent, tokenizer = load_checkpoint(ckpt_path, tok_path, device_str=args.device)
    agent.train()

    # Prior = deep copy of agent weights, frozen forever
    prior = copy.deepcopy(agent)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)

    # ── Load backbone model ───────────────────────────────────────────────────
    logger.info("Loading backbone EGFR model ...")
    backbone = QSARTrainer.load(_MODEL_DIR, cfg)

    # ── Load training SMILES ──────────────────────────────────────────────────
    logger.info("Loading training SMILES for novelty / AD fitting ...")
    train_smiles = _load_train_smiles()
    logger.info(f"  {len(train_smiles)} reference molecules")

    # ── Build applicability domain ────────────────────────────────────────────
    ad = ApplicabilityDomain.from_config()
    ad.fit(list(train_smiles))

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = MoleculeReward(
        backbone_model=backbone,
        ad=ad,
        train_smiles=train_smiles,
        cfg=reward_cfg,
    )

    # ── Pre-RL evaluation ─────────────────────────────────────────────────────
    logger.info(
        f"Pre-RL evaluation: sampling {eval_n} molecules at temp={eval_temp} ..."
    )
    t0 = time.time()
    pre_metrics = evaluate_generator(
        agent,
        tokenizer,
        train_smiles,
        ad,
        backbone,
        n=eval_n,
        temperature=eval_temp,
        device=device,
    )
    logger.info(f"  Pre-RL eval done in {time.time()-t0:.0f}s")
    _print_metrics("PRE-RL", pre_metrics)

    # ── RL training loop ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    scaffold_memory = None
    if df_cfg.get("enabled", False):
        scaffold_memory = ScaffoldMemory(
            bucket_size=int(df_cfg.get("bucket_size", 25)),
            min_score=float(df_cfg.get("min_score", 0.30)),
            penalty=float(df_cfg.get("penalty", 0.0)),
        )

    trainer = REINVENTTrainer(
        agent=agent,
        prior=prior,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        optimizer=optimizer,
        sigma=sigma,
        device=device,
        max_len=max_len,
        temperature=temp,
        scaffold_memory=scaffold_memory,
    )

    _LOGS_DIR.mkdir(exist_ok=True)
    step_log: list[dict] = []

    logger.info(f"\nStarting RL training: {n_steps} steps × batch_size={batch_sz} ...")
    t_train = time.time()

    with open(_LOG_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "step",
                "loss",
                "mean_reward",
                "mean_reward_raw",
                "n_penalized",
                "mean_nll_agent",
                "mean_nll_prior",
                "validity",
            ],
        )
        writer.writeheader()

        for step in range(1, n_steps + 1):
            step_stats = trainer.train_step(batch_size=batch_sz)
            step_stats["step"] = step
            writer.writerow(
                {
                    k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in step_stats.items()
                }
            )
            csvfile.flush()
            step_log.append(step_stats)

            if step % 10 == 0 or step == 1:
                logger.info(
                    f"  step {step:4d}/{n_steps}  "
                    f"loss={step_stats['loss']:.4f}  "
                    f"reward={step_stats['mean_reward']:.4f}  "
                    f"raw={step_stats['mean_reward_raw']:.4f}  "
                    f"penalized={step_stats['n_penalized']:2d}  "
                    f"validity={step_stats['validity']:.1%}"
                )

    elapsed = time.time() - t_train
    logger.info(f"RL training complete in {elapsed:.0f}s ({elapsed/n_steps:.1f}s/step)")
    if scaffold_memory is not None:
        logger.info(
            f"Scaffold memory: {scaffold_memory.n_scaffolds} distinct scaffolds seen, "
            f"{scaffold_memory.n_saturated} saturated, "
            f"{scaffold_memory.n_penalized} molecules penalised total"
        )

    # ── Save RL checkpoint ────────────────────────────────────────────────────
    agent.eval()
    rl_ckpt_path = _GEN_DIR / "rl_finetuned_gru.pt"
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "model_config": agent.config(),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "rl_n_steps": n_steps,
            "rl_batch_size": batch_sz,
            "rl_sigma": sigma,
            "rl_lr": lr,
            "base_ckpt": str(ckpt_path),
        },
        rl_ckpt_path,
    )
    logger.info(f"RL checkpoint saved to {rl_ckpt_path}")

    # ── Post-RL evaluation ────────────────────────────────────────────────────
    logger.info(
        f"\nPost-RL evaluation: sampling {eval_n} molecules at temp={eval_temp} ..."
    )
    t0 = time.time()
    post_metrics = evaluate_generator(
        agent,
        tokenizer,
        train_smiles,
        ad,
        backbone,
        n=eval_n,
        temperature=eval_temp,
        device=device,
    )
    logger.info(f"  Post-RL eval done in {time.time()-t0:.0f}s")
    _print_metrics("POST-RL", post_metrics)

    # ── Pre-vs-post comparison ────────────────────────────────────────────────
    comparison = compare_pre_post(pre_metrics, post_metrics)

    print("\n" + "=" * 60)
    print("PRE-vs-POST RL COMPARISON")
    print("=" * 60)
    print(comparison["table"])
    print(f"\n  VERDICT: {comparison['verdict']}")
    print(f"  Detail : {comparison['detail']}")
    print("=" * 60)

    if comparison["verdict"] == "REWARD_HACKING":
        print("\n  *** REWARD HACKING DETECTED ***")
        print("  Recommend: reduce sigma, increase OOD/diversity penalties,")
        print("  or reduce n_steps. Roll back to egfr_finetuned_gru.pt.")

    # ── Save results JSON ─────────────────────────────────────────────────────
    diversity_filter_info: dict = {"enabled": scaffold_memory is not None}
    if scaffold_memory is not None:
        diversity_filter_info.update(
            {
                "bucket_size": scaffold_memory.bucket_size,
                "min_score": scaffold_memory.min_score,
                "penalty": scaffold_memory.penalty,
                "n_scaffolds_seen": scaffold_memory.n_scaffolds,
                "n_scaffolds_saturated": scaffold_memory.n_saturated,
                "n_molecules_penalized": scaffold_memory.n_penalized,
            }
        )

    results = {
        "n_steps": n_steps,
        "batch_size": batch_sz,
        "sigma": sigma,
        "lr": lr,
        "temperature": temp,
        "eval_n": eval_n,
        "diversity_filter": diversity_filter_info,
        "pre_rl": pre_metrics,
        "post_rl": post_metrics,
        "comparison": {
            "verdict": comparison["verdict"],
            "detail": comparison["detail"],
            "table": comparison["table"],
        },
        "step_log": step_log[-50:],  # last 50 steps (tail of training)
        "log_csv": str(_LOG_CSV),
        "checkpoint": str(rl_ckpt_path),
        "note": (
            "EXPLORATORY. RL-tuned GRU, REINVENT augmented-NLL objective with "
            "scaffold-memory diversity filter. Backbone pIC50 is in-sample; AD "
            "guards against OOD reward hacking; scaffold memory guards against "
            "mode collapse. Apply docking post-hoc to top-ranked hits; do not use "
            "raw pIC50 as ground truth."
        ),
    }
    out_path = _GEN_DIR / "rl_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
