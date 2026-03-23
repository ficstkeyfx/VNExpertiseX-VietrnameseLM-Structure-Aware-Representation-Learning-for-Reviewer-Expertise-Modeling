"""
Ablation Study for ExpertiseX-LM  (Section 3.5)

Five configurations compared on the SAME test set:

  1. Full model (ExpertiseX-LM)         — all components enabled
  2. w/o Expertise Embeddings           — expertise_positions zeroed at inference
  3. w/o MSLM pre-training              — model fine-tuned from PhoBERT directly
  4. w/o Research Area nodes            — area nodes omitted during serialization
  5. w/o Negative sampling              — model trained on positive-only data

Configs 1, 2, 4 only need the full-model checkpoint.
Config 3 needs a separate model (--no_pretrain_model_dir) trained by running
    finetune.py --pretrained_model_dir /nonexistent
    (this forces PhoBERT→finetune without MSLM).
    If omitted, raw PhoBERT+random-classifier is used as a rough lower-bound.
Config 5 needs a separate model (--no_negative_model_dir) trained on
    positive-only data.  If omitted, this ablation is skipped.

Usage
-----
    python eval_ablation.py --model_dir ../train/finetune_output/final
    python eval_ablation.py --model_dir ../train/finetune_output/final \
        --no_pretrain_model_dir ../train/finetune_nopretrain/final   \
        --no_negative_model_dir ../train/finetune_noneg/final
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from dataset import ExpertiseGraphDataset
from model import ExpertiseXLMForSequenceClassification

from eval_topk import (
    set_seed,
    load_model,
    build_field_to_authors,
    build_candidate_pool,
    evaluate,
)


# ───────────────────────── main ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study for ExpertiseX-LM (Section 3.5)"
    )
    parser.add_argument("--data_dir", type=str,
                        default=r"d:\C500\Lab306\Reviewer_Recommendation\crawl-data\articles")
    parser.add_argument("--model_dir", type=str,
                        default=r"..\train\finetune_output\final",
                        help="Full-model checkpoint (MSLM-pretrained → fine-tuned)")
    parser.add_argument("--no_pretrain_model_dir", type=str, default=None,
                        help="Model fine-tuned directly from PhoBERT (no MSLM). "
                             "If omitted, PhoBERT + random classifier is used.")
    parser.add_argument("--no_negative_model_dir", type=str, default=None,
                        help="Model trained without negative sampling. "
                             "If omitted, this ablation row is skipped.")
    parser.add_argument("--top_k",            type=int,   default=5)
    parser.add_argument("--num_test_papers",  type=int,   default=50)
    parser.add_argument("--num_candidates",   type=int,   default=200)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output_csv",       type=str,   default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── shared data (loaded once) ───────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "[TARGET_PAPER]", "[REVIEWER_PAPER]", "[RESEARCH_AREA]"
        ]
    })

    dataset = ExpertiseGraphDataset(
        data_dir=args.data_dir, eval_csv=None,
        tokenizer=tokenizer, is_pretrain=True,
    )

    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.vocab_size = len(tokenizer)
    config.num_labels = 1

    field_to_authors = build_field_to_authors(dataset)
    all_candidates   = build_candidate_pool(dataset)
    print(f"Total authors in pool: {len(all_candidates)}")

    # Fixed test set so every config is compared fairly
    test_ids = random.sample(
        dataset.all_paper_ids,
        min(args.num_test_papers, len(dataset.all_paper_ids)),
    )

    # ── ablation registry ───────────────────────────────────
    #   key  : display name
    #   value: dict with model_source + eval kwargs
    ABLATIONS = {}

    ABLATIONS["Full model (ExpertiseX-LM)"] = dict(
        model_source="checkpoint", model_dir=args.model_dir,
        use_expertise=True, use_areas=True,
    )
    ABLATIONS["w/o Expertise Embeddings"] = dict(
        model_source="checkpoint", model_dir=args.model_dir,
        use_expertise=False, use_areas=True,
    )

    if args.no_pretrain_model_dir and os.path.isdir(args.no_pretrain_model_dir):
        ABLATIONS["w/o MSLM pre-training"] = dict(
            model_source="checkpoint", model_dir=args.no_pretrain_model_dir,
            use_expertise=True, use_areas=True,
        )
    else:
        ABLATIONS["w/o MSLM pre-training"] = dict(
            model_source="phobert_base",
            use_expertise=True, use_areas=True,
        )

    ABLATIONS["w/o Research Area nodes"] = dict(
        model_source="checkpoint", model_dir=args.model_dir,
        use_expertise=True, use_areas=False,
    )

    if args.no_negative_model_dir and os.path.isdir(args.no_negative_model_dir):
        ABLATIONS["w/o Negative sampling"] = dict(
            model_source="checkpoint", model_dir=args.no_negative_model_dir,
            use_expertise=True, use_areas=True,
        )

    # ── run each ablation ───────────────────────────────────
    rows = []

    for name, cfg in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f" ABLATION: {name}")
        print(f"{'='*60}")

        # Reset seed so candidate sampling is identical across configs
        set_seed(args.seed)

        # Load model for this config
        if cfg["model_source"] == "phobert_base":
            print("  (PhoBERT base + random classifier — lower-bound proxy)")
            model = ExpertiseXLMForSequenceClassification.from_pretrained_phobert(
                "vinai/phobert-base-v2", config
            )
            model.to(device)
            model.eval()
        else:
            model = load_model(cfg["model_dir"], config, device)

        results, _ = evaluate(
            model, dataset, test_ids, all_candidates,
            field_to_authors, device,
            k=args.top_k, n_candidates=args.num_candidates,
            batch_size=args.batch_size,
            use_expertise=cfg["use_expertise"],
            use_areas=cfg["use_areas"],
            verbose=False,
        )

        results["Configuration"] = name
        rows.append(results)

        for metric, val in results.items():
            if metric != "Configuration":
                print(f"  {metric:15s}: {val:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── comparison table ────────────────────────────────────
    df = pd.DataFrame(rows)
    cols = ["Configuration"] + [c for c in df.columns if c != "Configuration"]
    df = df[cols]

    print(f"\n{'='*70}")
    print(" ABLATION COMPARISON TABLE")
    print(f"{'='*70}")
    print(df.to_string(index=False, float_format="%.4f"))

    out = args.output_csv or os.path.join(
        os.path.dirname(__file__), "ablation_results.csv"
    )
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
