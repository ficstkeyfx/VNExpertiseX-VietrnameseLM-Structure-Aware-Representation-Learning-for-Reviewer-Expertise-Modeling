"""
Top-K Reviewer Recommendation Evaluation  (Section 3.2 – 3.4)

Protocol
--------
- Candidate pool : N reviewers randomly sampled per test paper (default 200).
- Ground truth   : Author-as-Reviewer  +  Same-Field,  COI-filtered.
- Ranking        : Model scores every (paper, reviewer) pair; top-K extracted.
- Metrics (7)    : P@K, R@K, NDCG@K, MAP, MRR, AUC-ROC, F1-Score.

Usage
-----
    python eval_topk.py --model_dir ../train/finetune_output/final
    python eval_topk.py --num_test_papers 50 --num_candidates 200 --top_k 5
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

from metrics import compute_query_metrics, aggregate_metrics


# ───────────────────────── helpers ──────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_dir, config, device):
    """Load ExpertiseXLMForSequenceClassification from a checkpoint dir."""
    model = ExpertiseXLMForSequenceClassification(config)
    model_bin = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(model_bin):
        state_dict = torch.load(model_bin, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights from {model_bin}")
    elif os.path.isdir(model_dir):
        try:
            model = ExpertiseXLMForSequenceClassification.from_pretrained(
                model_dir, config=config
            )
            print(f"  Loaded from HF directory {model_dir}")
        except Exception as e:
            print(f"  Warning: could not load weights ({e}), using random init")
    else:
        print(f"  Warning: {model_dir} not found — using random weights")

    model.to(device)
    model.eval()
    return model


def build_field_to_authors(dataset):
    """research_field → set(authors)"""
    mapping = {}
    for author, papers in dataset.author_to_papers.items():
        for pid in papers:
            for area in dataset.research_areas.get(pid, []):
                mapping.setdefault(area, set()).add(author)
    return mapping


def build_candidate_pool(dataset):
    """Full candidate pool: one entry per known author."""
    return [
        {"id": author, "authored_papers": set(papers)}
        for author, papers in dataset.author_to_papers.items()
    ]


# ───────────────────── ground truth ─────────────────────────

def get_ground_truth(dataset, target_paper_id, field_to_authors):
    """
    Dual-strategy ground truth (Section 3.2):
      Strategy 1 — Author-as-Reviewer  (co-authors of the paper)
      Strategy 2 — Same-Field          (all authors in the same research field)
      COI filter  — actual authors removed from gt_relevant

    Returns (gt_relevant, actual_authors).
    """
    target_paper = dataset._read_paper(target_paper_id)
    if not target_paper:
        return set(), set()

    actual_authors = set()
    author_str = target_paper.get("author", "")
    if author_str:
        for a in author_str.split(","):
            a = a.strip()
            if a:
                actual_authors.add(a)

    gt_authors = set(actual_authors)

    for area in dataset.research_areas.get(target_paper_id, []):
        gt_authors.update(field_to_authors.get(area, set()))

    gt_relevant = gt_authors - actual_authors
    return gt_relevant, actual_authors


# ───────────────── batched model scoring ────────────────────

def score_candidates(
    model, dataset, target_paper, candidate_reviewers, target_paper_id,
    device, batch_size=64,
    use_expertise=True, use_areas=True,
):
    """
    Score every candidate against *target_paper* with batched inference.

    Ablation toggles
    ----------------
    use_expertise : if False → expertise_positions zeroed (standard position only)
    use_areas     : if False → research-area nodes omitted from serialization
    """
    target_areas = (
        dataset.research_areas.get(target_paper_id, []) if use_areas else []
    )

    candidate_features, candidate_ids = [], []

    for reviewer in candidate_reviewers:
        rid = reviewer["id"]
        if target_paper_id in reviewer["authored_papers"]:
            continue

        participant = dataset._read_participant(rid)
        rev_papers = []
        if participant and "papers" in participant:
            for p in participant["papers"][:5]:
                p_data = dataset._read_paper(p["paperId"])
                if p_data:
                    rev_papers.append(p_data)

        features = dataset._serialize_subgraph(
            target_paper, rev_papers, target_areas, label=0.0
        )

        if not use_expertise:
            features["expertise_positions"] = torch.zeros_like(
                features["expertise_positions"]
            )

        candidate_features.append(features)
        candidate_ids.append(rid)

    if not candidate_features:
        return [], []

    all_scores = []
    with torch.no_grad():
        for start in range(0, len(candidate_features), batch_size):
            batch = candidate_features[start : start + batch_size]
            b_ids  = torch.stack([f["input_ids"]           for f in batch]).to(device)
            b_mask = torch.stack([f["attention_mask"]       for f in batch]).to(device)
            b_pos  = torch.stack([f["expertise_positions"]  for f in batch]).to(device)

            logits = model(
                input_ids=b_ids,
                attention_mask=b_mask,
                expertise_positions=b_pos,
            ).logits.squeeze(-1)

            scores = logits.cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

    return candidate_ids, all_scores


# ───────────────────── main evaluation ──────────────────────

def evaluate(
    model, dataset, test_paper_ids, all_candidates, field_to_authors,
    device,
    k=5, n_candidates=200, batch_size=64,
    use_expertise=True, use_areas=True,
    verbose=True,
):
    """
    Full Top-K evaluation loop.

    Returns
    -------
    results        : dict   — aggregated metric values
    per_query      : list   — per-paper metric dicts
    """
    per_query = []

    if verbose:
        print(f"\n{'='*60}")
        print(f" Top-{k} Reviewer Recommendation Evaluation")
        print(f" Test papers: {len(test_paper_ids)}, Candidate pool size N={n_candidates}")
        print(f"{'='*60}")

    for idx, paper_id in enumerate(test_paper_ids):
        target_paper = dataset._read_paper(paper_id)
        if not target_paper:
            continue

        gt_relevant, _ = get_ground_truth(dataset, paper_id, field_to_authors)

        # Sample candidate pool (Section 3.2)
        if 0 < n_candidates < len(all_candidates):
            candidates = random.sample(all_candidates, n_candidates)
        else:
            candidates = all_candidates

        cand_ids, scores = score_candidates(
            model, dataset, target_paper, candidates, paper_id,
            device, batch_size, use_expertise, use_areas,
        )
        if not cand_ids:
            continue

        # Rank by descending score
        paired = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
        ranked_ids = [p[0] for p in paired]

        y_true   = [1 if cid in gt_relevant else 0 for cid in cand_ids]
        y_scores = scores

        qm = compute_query_metrics(ranked_ids, gt_relevant, y_true, y_scores, k)
        per_query.append(qm)

        if verbose:
            title = target_paper.get("title", "N/A")[:80]
            pk  = qm[f"Precision@{k}"]
            rk  = qm[f"Recall@{k}"]
            nk  = qm[f"NDCG@{k}"]
            ap  = qm["MAP"]
            rr  = qm["MRR"]
            print(f"\n[{idx+1}/{len(test_paper_ids)}] {paper_id}")
            print(f"  Title : {title}...")
            print(f"  GT relevant reviewers: {len(gt_relevant)}")
            print(f"  P@{k}={pk:.4f}  R@{k}={rk:.4f}  NDCG@{k}={nk:.4f}  "
                  f"AP={ap:.4f}  RR={rr:.4f}")
            print(f"  Top-{k}:")
            for rank, (rid, sc) in enumerate(paired[:k], 1):
                tag = " [RELEVANT]" if rid in gt_relevant else ""
                print(f"    {rank}. {rid}  (score {sc:.4f}){tag}")

    results = aggregate_metrics(per_query, k)

    if verbose:
        print(f"\n{'='*60}")
        print(f" AGGREGATE ({len(per_query)} test papers)")
        print(f"{'='*60}")
        for name, val in results.items():
            print(f"  {name:15s}: {val:.4f}")

    return results, per_query


# ────────────────────────── CLI ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Top-K Reviewer Recommendation Evaluation (Section 3.2–3.4)"
    )
    parser.add_argument("--data_dir",    type=str, default=r"d:\C500\Lab306\Reviewer_Recommendation\crawl-data\articles")
    parser.add_argument("--model_dir",   type=str, default=r"..\train\finetune_output\final")
    parser.add_argument("--top_k",       type=int, default=5,   help="K for Top-K recommendation")
    parser.add_argument("--num_test_papers",  type=int, default=50, help="Number of test papers")
    parser.add_argument("--num_candidates",   type=int, default=200, help="Candidate pool size N")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_csv",  type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── tokenizer + dataset ──
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["[TARGET_PAPER]", "[REVIEWER_PAPER]", "[RESEARCH_AREA]"]
    })

    dataset = ExpertiseGraphDataset(
        data_dir=args.data_dir, eval_csv=None,
        tokenizer=tokenizer, is_pretrain=True,
    )

    # ── model ──
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.vocab_size = len(tokenizer)
    config.num_labels = 1

    print(f"\nLoading model from: {args.model_dir}")
    model = load_model(args.model_dir, config, device)

    # ── evaluation data ──
    field_to_authors = build_field_to_authors(dataset)
    all_candidates   = build_candidate_pool(dataset)
    print(f"Total authors in pool: {len(all_candidates)}")

    test_ids = random.sample(
        dataset.all_paper_ids,
        min(args.num_test_papers, len(dataset.all_paper_ids)),
    )

    # ── run ──
    results, per_query = evaluate(
        model, dataset, test_ids, all_candidates,
        field_to_authors, device,
        k=args.top_k, n_candidates=args.num_candidates,
        batch_size=args.batch_size,
    )

    # ── save ──
    out = args.output_csv or os.path.join(os.path.dirname(__file__), "eval_topk_results.csv")
    pd.DataFrame([results]).to_csv(out, index=False)
    print(f"\nAggregate results → {out}")

    per_q_out = out.replace(".csv", "_per_query.csv")
    pd.DataFrame(per_query).to_csv(per_q_out, index=False)
    print(f"Per-query results  → {per_q_out}")


if __name__ == "__main__":
    main()
