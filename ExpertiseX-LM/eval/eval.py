import os
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import roc_auc_score, f1_score
import sys

# To import from train folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from dataset import ExpertiseGraphDataset
from model import ExpertiseXLMForSequenceClassification

# ======================== METRICS ========================

def precision_at_k(ranked_list, relevant_set, k):
    """Proportion of top-k items that are relevant."""
    top_k = ranked_list[:k]
    return len([r for r in top_k if r in relevant_set]) / k

def recall_at_k(ranked_list, relevant_set, k):
    """Proportion of relevant items found in top-k."""
    if len(relevant_set) == 0:
        return 0.0
    top_k = ranked_list[:k]
    return len([r for r in top_k if r in relevant_set]) / len(relevant_set)

def dcg_at_k(ranked_list, relevant_set, k):
    """Discounted Cumulative Gain."""
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        rel = 1.0 if item in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg

def ndcg_at_k(ranked_list, relevant_set, k):
    """Normalized DCG."""
    dcg = dcg_at_k(ranked_list, relevant_set, k)
    # Ideal ranking: all relevant items first
    ideal_list = list(relevant_set)[:k] + [None] * max(0, k - len(relevant_set))
    idcg = dcg_at_k(ideal_list, relevant_set, k)
    return dcg / idcg if idcg > 0 else 0.0

def average_precision(ranked_list, relevant_set):
    """Average Precision for one query."""
    if len(relevant_set) == 0:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(ranked_list):
        if item in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant_set)

def reciprocal_rank(ranked_list, relevant_set):
    """Reciprocal Rank: 1/position of first relevant item."""
    for i, item in enumerate(ranked_list):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"d:\C500\Lab306\Reviewer_Recommendation\crawl-data\articles")
    parser.add_argument("--model_dir", type=str, default=r"..\train\finetune_output\final")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_test_papers", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=200)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    special_tokens_dict = {'additional_special_tokens': ['[TARGET_PAPER]', '[REVIEWER_PAPER]', '[RESEARCH_AREA]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    dataset = ExpertiseGraphDataset(
        data_dir=args.data_dir,
        eval_csv="dummy.csv",
        tokenizer=tokenizer,
        is_pretrain=False
    )
    dataset.pairs = []
    
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.vocab_size = len(tokenizer)
    config.num_labels = 1
    
    model = ExpertiseXLMForSequenceClassification(config)
    if os.path.exists(args.model_dir):
        print(f"Loading model from {args.model_dir}...")
        try:
            model = ExpertiseXLMForSequenceClassification.from_pretrained(args.model_dir, config=config)
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
    else:
        print("Pre-trained fine-tuning model not found, using random weights for demonstration.")

    model.eval()
    
    # Build field-to-authors mapping for Same-Field ground truth
    field_to_authors = {}
    for author, papers in dataset.author_to_papers.items():
        for pid in papers:
            areas = dataset.research_areas.get(pid, [])
            for area in areas:
                if area not in field_to_authors:
                    field_to_authors[area] = set()
                field_to_authors[area].add(author)
    
    # Candidate pool
    all_candidates = []
    for author, papers in dataset.author_to_papers.items():
        all_candidates.append({"id": author, "authored_papers": papers})
    
    print(f"Total candidate reviewers: {len(all_candidates)}")
    
    # Limit candidates
    if args.num_candidates > 0 and args.num_candidates < len(all_candidates):
        candidate_reviewers = random.sample(all_candidates, args.num_candidates)
    else:
        candidate_reviewers = all_candidates
    print(f"Using {len(candidate_reviewers)} candidate reviewers for evaluation.")
    
    # Sample test papers
    test_paper_ids = random.sample(dataset.all_paper_ids, min(args.num_test_papers, len(dataset.all_paper_ids)))
    
    # Aggregate metrics
    all_precisions = []
    all_recalls = []
    all_ndcgs = []
    all_aps = []
    all_rrs = []
    all_auc = []
    all_f1 = []
    
    K = args.top_k
    
    print(f"\n{'='*60}")
    print(f" EVALUATION: Top-{K} Reviewer Recommendation")
    print(f" Test papers: {len(test_paper_ids)}, Candidate pool: {len(candidate_reviewers)}")
    print(f"{'='*60}")
    
    with torch.no_grad():
        for paper_idx, target_paper_id in enumerate(test_paper_ids):
            target_paper = dataset._read_paper(target_paper_id)
            if not target_paper:
                continue
            
            # Ground truth: authors of this paper + same-field authors
            gt_authors = set()
            # Strategy 1: Author-as-Reviewer
            author_str = target_paper.get("author", "")
            if author_str:
                for a in author_str.split(","):
                    a = a.strip()
                    if a:
                        gt_authors.add(a)
            
            # Strategy 2: Same-Field authors
            target_areas = dataset.research_areas.get(target_paper_id, [])
            for area in target_areas:
                same_field = field_to_authors.get(area, set())
                gt_authors.update(same_field)
            
            # Remove actual authors from gt for COI (they can't review their own paper)
            actual_authors = set()
            if author_str:
                for a in author_str.split(","):
                    a = a.strip()
                    if a:
                        actual_authors.add(a)
            gt_relevant = gt_authors - actual_authors
            
            predictions = []
            y_true = []
            y_scores = []
            
            for reviewer in candidate_reviewers:
                reviewer_id = reviewer["id"]
                
                # COI filter
                if target_paper_id in reviewer["authored_papers"]:
                    continue
                
                participant_data = dataset._read_participant(reviewer_id)
                reviewer_papers = []
                if participant_data and "papers" in participant_data:
                    for p in participant_data["papers"][:5]:
                        p_data = dataset._read_paper(p["paperId"])
                        if p_data:
                            reviewer_papers.append(p_data)
                        
                target_areas_list = dataset.research_areas.get(target_paper_id, [])
                
                features = dataset._serialize_subgraph(target_paper, reviewer_papers, target_areas_list, label=0.0)
                input_ids = features['input_ids'].unsqueeze(0)
                attention_mask = features['attention_mask'].unsqueeze(0)
                expertise_positions = features['expertise_positions'].unsqueeze(0)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    expertise_positions=expertise_positions
                )
                
                score = outputs.logits.item()
                predictions.append((reviewer_id, score))
                
                # Binary labels for AUC/F1
                is_relevant = 1 if reviewer_id in gt_relevant else 0
                y_true.append(is_relevant)
                y_scores.append(score)
            
            if not predictions:
                continue
            
            # Rank by score descending
            predictions.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [p[0] for p in predictions]
            
            # Calculate metrics
            p_at_k = precision_at_k(ranked_ids, gt_relevant, K)
            r_at_k = recall_at_k(ranked_ids, gt_relevant, K)
            n_at_k = ndcg_at_k(ranked_ids, gt_relevant, K)
            ap = average_precision(ranked_ids, gt_relevant)
            rr = reciprocal_rank(ranked_ids, gt_relevant)
            
            all_precisions.append(p_at_k)
            all_recalls.append(r_at_k)
            all_ndcgs.append(n_at_k)
            all_aps.append(ap)
            all_rrs.append(rr)
            
            # AUC-ROC (needs both classes)
            if len(set(y_true)) > 1:
                auc = roc_auc_score(y_true, y_scores)
                all_auc.append(auc)
            
            # F1 (threshold at median score)
            if y_scores:
                threshold = np.median(y_scores)
                y_pred = [1 if s >= threshold else 0 for s in y_scores]
                if len(set(y_pred)) > 1:
                    f1 = f1_score(y_true, y_pred)
                    all_f1.append(f1)
            
            # Print per-paper results
            print(f"\n[{paper_idx+1}/{len(test_paper_ids)}] Paper: {target_paper_id}")
            print(f"  Title: {target_paper.get('title', 'N/A')[:80]}...")
            print(f"  GT relevant reviewers: {len(gt_relevant)}")
            print(f"  P@{K}={p_at_k:.4f}  R@{K}={r_at_k:.4f}  NDCG@{K}={n_at_k:.4f}  AP={ap:.4f}  RR={rr:.4f}")
            print(f"  Top-{K} recommendations:")
            for rank, (rev_id, score) in enumerate(predictions[:K], 1):
                marker = " ✓" if rev_id in gt_relevant else ""
                print(f"    {rank}. {rev_id} (score: {score:.4f}){marker}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f" AGGREGATE RESULTS (over {len(all_precisions)} test papers)")
    print(f"{'='*60}")
    
    results = {
        f"Precision@{K}": np.mean(all_precisions) if all_precisions else 0,
        f"Recall@{K}": np.mean(all_recalls) if all_recalls else 0,
        f"NDCG@{K}": np.mean(all_ndcgs) if all_ndcgs else 0,
        "MAP": np.mean(all_aps) if all_aps else 0,
        "MRR": np.mean(all_rrs) if all_rrs else 0,
        "AUC-ROC": np.mean(all_auc) if all_auc else 0,
        "F1-Score": np.mean(all_f1) if all_f1 else 0,
    }
    
    for metric, value in results.items():
        print(f"  {metric:15s}: {value:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()

