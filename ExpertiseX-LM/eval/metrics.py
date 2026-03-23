"""
Evaluation metrics for Top-K Reviewer Recommendation.

Implements 7 standard IR/classification metrics:
  1. Precision@K
  2. Recall@K
  3. NDCG@K
  4. MAP (Mean Average Precision)
  5. MRR (Mean Reciprocal Rank)
  6. AUC-ROC
  7. F1-Score (median-threshold binarization)
"""

import math
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def precision_at_k(ranked_ids, relevant_set, k):
    """P@K = |relevant ∩ top-K| / K"""
    top_k = ranked_ids[:k]
    return sum(1 for r in top_k if r in relevant_set) / k


def recall_at_k(ranked_ids, relevant_set, k):
    """R@K = |relevant ∩ top-K| / |relevant|"""
    if not relevant_set:
        return 0.0
    top_k = ranked_ids[:k]
    return sum(1 for r in top_k if r in relevant_set) / len(relevant_set)


def dcg_at_k(ranked_ids, relevant_set, k):
    """DCG@K = Σ rel_i / log2(i+1)"""
    dcg = 0.0
    for i, item in enumerate(ranked_ids[:k]):
        rel = 1.0 if item in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_ids, relevant_set, k):
    """NDCG@K = DCG@K / IDCG@K"""
    dcg = dcg_at_k(ranked_ids, relevant_set, k)
    ideal = list(relevant_set)[:k] + [None] * max(0, k - len(relevant_set))
    idcg = dcg_at_k(ideal, relevant_set, k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked_ids, relevant_set):
    """AP = (1/|relevant|) Σ (hits/rank) for each relevant item."""
    if not relevant_set:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(ranked_ids):
        if item in relevant_set:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant_set)


def reciprocal_rank(ranked_ids, relevant_set):
    """RR = 1 / rank of first relevant item."""
    for i, item in enumerate(ranked_ids):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_auc(y_true, y_scores):
    """AUC-ROC. Returns None if only one class present."""
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_scores)


def compute_f1(y_true, y_scores):
    """F1 using median score as decision threshold for binarization."""
    if not y_scores:
        return None
    threshold = np.median(y_scores)
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    if len(set(y_pred)) < 2 or len(set(y_true)) < 2:
        return None
    return f1_score(y_true, y_pred)


def compute_query_metrics(ranked_ids, gt_relevant, y_true, y_scores, k):
    """Compute all 7 metrics for a single query (one test paper)."""
    return {
        f"Precision@{k}": precision_at_k(ranked_ids, gt_relevant, k),
        f"Recall@{k}": recall_at_k(ranked_ids, gt_relevant, k),
        f"NDCG@{k}": ndcg_at_k(ranked_ids, gt_relevant, k),
        "MAP": average_precision(ranked_ids, gt_relevant),
        "MRR": reciprocal_rank(ranked_ids, gt_relevant),
        "AUC-ROC": compute_auc(y_true, y_scores),
        "F1-Score": compute_f1(y_true, y_scores),
    }


def aggregate_metrics(all_query_metrics, k):
    """Aggregate per-query metric dicts into means (ignoring None values)."""
    metric_names = [
        f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}",
        "MAP", "MRR", "AUC-ROC", "F1-Score",
    ]
    results = {}
    for name in metric_names:
        values = [m[name] for m in all_query_metrics if m.get(name) is not None]
        results[name] = float(np.mean(values)) if values else 0.0
    return results
