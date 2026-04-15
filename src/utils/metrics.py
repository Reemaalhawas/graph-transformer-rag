"""
Retrieval evaluation metrics used throughout the project.

All functions accept:
  - ranked_lists : list of lists — ranked candidate node-id lists per query
  - ground_truth : list of lists — correct answer node-id lists per query

Metrics implemented:
  - Hits@K        (fraction of queries with ≥1 correct answer in top-K)
  - MRR           (Mean Reciprocal Rank)
  - MAP           (Mean Average Precision)
  - Precision@K
  - Recall@K
"""

from typing import List, Dict
import numpy as np


def hits_at_k(ranked_lists: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """Fraction of queries where at least one correct answer is in the top-K results."""
    hits = 0
    for ranked, gt in zip(ranked_lists, ground_truth):
        gt_set = set(gt)
        if any(node in gt_set for node in ranked[:k]):
            hits += 1
    return hits / len(ranked_lists) if ranked_lists else 0.0


def mean_reciprocal_rank(ranked_lists: List[List[int]], ground_truth: List[List[int]]) -> float:
    """Mean Reciprocal Rank (MRR) across all queries."""
    rr_sum = 0.0
    for ranked, gt in zip(ranked_lists, ground_truth):
        gt_set = set(gt)
        for rank, node in enumerate(ranked, start=1):
            if node in gt_set:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(ranked_lists) if ranked_lists else 0.0


def mean_average_precision(ranked_lists: List[List[int]], ground_truth: List[List[int]]) -> float:
    """Mean Average Precision (MAP) across all queries."""
    ap_sum = 0.0
    for ranked, gt in zip(ranked_lists, ground_truth):
        gt_set = set(gt)
        if not gt_set:
            continue
        num_correct = 0
        precision_sum = 0.0
        for rank, node in enumerate(ranked, start=1):
            if node in gt_set:
                num_correct += 1
                precision_sum += num_correct / rank
        ap_sum += precision_sum / len(gt_set)
    return ap_sum / len(ranked_lists) if ranked_lists else 0.0


def precision_at_k(ranked_lists: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """Average Precision@K across all queries."""
    p_sum = 0.0
    for ranked, gt in zip(ranked_lists, ground_truth):
        gt_set = set(gt)
        top_k = ranked[:k]
        p_sum += len([n for n in top_k if n in gt_set]) / k
    return p_sum / len(ranked_lists) if ranked_lists else 0.0


def recall_at_k(ranked_lists: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """Average Recall@K across all queries."""
    r_sum = 0.0
    for ranked, gt in zip(ranked_lists, ground_truth):
        gt_set = set(gt)
        if not gt_set:
            continue
        top_k = set(ranked[:k])
        r_sum += len(top_k & gt_set) / len(gt_set)
    return r_sum / len(ranked_lists) if ranked_lists else 0.0


def compute_metrics(
    ranked_lists: List[List[int]],
    ground_truth: List[List[int]],
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    Compute the full set of retrieval metrics.

    Returns a dict like:
    {
        "hits@1": 0.xx, "hits@5": 0.xx, "hits@10": 0.xx,
        "mrr": 0.xx,
        "map": 0.xx,
        "precision@10": 0.xx,
        "recall@10": 0.xx,
    }
    """
    metrics: Dict[str, float] = {}

    for k in k_values:
        metrics[f"hits@{k}"]      = hits_at_k(ranked_lists, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked_lists, ground_truth, k)
        metrics[f"recall@{k}"]    = recall_at_k(ranked_lists, ground_truth, k)

    metrics["mrr"] = mean_reciprocal_rank(ranked_lists, ground_truth)
    metrics["map"] = mean_average_precision(ranked_lists, ground_truth)

    return metrics
