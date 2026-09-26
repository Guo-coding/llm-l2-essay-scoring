#!/usr/bin/env python3
"""Summarise holistic and analytic ELLIPSE scoring results."""

import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from paths import ELLIPSE_RESULTS, RESULTS_DIR


# paths
INPUT_JSONL = ELLIPSE_RESULTS
OUTPUT_XLSX = RESULTS_DIR / "summary" / "ellipse_eval_summary.xlsx"


# score scale
SCORE_ORDER = ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"]
SCORE_TO_INT = {s: i for i, s in enumerate(SCORE_ORDER)}
INT_TO_SCORE = {i: s for s, i in SCORE_TO_INT.items()}

ADJACENT_TOLERANCE = 0.5
EXPECTED_RUNS = 5

DIMENSIONS = [
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
]


# score helpers
def clean_score(x):
    if x is None:
        return None

    if isinstance(x, (int, float)):
        x = f"{float(x):.1f}"
    elif isinstance(x, str):
        x = x.strip()
        try:
            x = f"{float(x):.1f}"
        except ValueError:
            return None
    else:
        return None

    return x if x in SCORE_TO_INT else None


def is_analytic_pred_list(pred_list):
    if not isinstance(pred_list, list) or len(pred_list) == 0:
        return False
    return all(isinstance(x, dict) for x in pred_list)


def extract_prediction_list(pred_list, target_dimension="Overall"):
    if not isinstance(pred_list, list) or len(pred_list) == 0:
        return []

    if all(not isinstance(x, dict) for x in pred_list):
        if target_dimension == "Overall":
            return pred_list[:]
        return []

    extracted = []
    for item in pred_list:
        if isinstance(item, dict):
            extracted.append(item.get(target_dimension))
    return extracted


def median_score(labels):
    values = sorted(float(label) for label in labels if label is not None)
    if len(values) != EXPECTED_RUNS:
        return None
    return clean_score(values[len(values) // 2])


def mean_prediction(labels):
    vals = [float(l) for l in labels if l is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def std_prediction(labels):
    vals = [float(l) for l in labels if l is not None]
    if len(vals) <= 1:
        return 0.0 if len(vals) == 1 else None

    mean_val = sum(vals) / len(vals)
    var = sum((x - mean_val) ** 2 for x in vals) / len(vals)
    return math.sqrt(var)


def adjacent_correct(pred, gold, tol=ADJACENT_TOLERANCE):
    if pred is None or gold is None:
        return False
    return abs(float(pred) - float(gold)) <= tol


def mean_absolute_error_single(pred, gold):
    if pred is None or gold is None:
        return None
    return abs(float(pred) - float(gold))


def signed_error_single(pred, gold):
    if pred is None or gold is None:
        return None
    return float(pred) - float(gold)


def quadratic_weighted_kappa(y_true_int, y_pred_int, n_classes):
    if len(y_true_int) == 0:
        return None

    O = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true_int, y_pred_int):
        O[t][p] += 1

    hist_true = [0] * n_classes
    hist_pred = [0] * n_classes
    for t in y_true_int:
        hist_true[t] += 1
    for p in y_pred_int:
        hist_pred[p] += 1

    N = len(y_true_int)

    E = [[0] * n_classes for _ in range(n_classes)]
    for i in range(n_classes):
        for j in range(n_classes):
            E[i][j] = (hist_true[i] * hist_pred[j]) / N

    W = [[0] * n_classes for _ in range(n_classes)]
    denom = (n_classes - 1) ** 2
    for i in range(n_classes):
        for j in range(n_classes):
            W[i][j] = ((i - j) ** 2) / denom

    num = 0.0
    den = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            num += W[i][j] * O[i][j]
            den += W[i][j] * E[i][j]

    if den == 0:
        return None

    return 1.0 - (num / den)


def stability_unanimous(labels):
    labels = [l for l in labels if l is not None]
    if not labels:
        return None
    return len(set(labels)) == 1


def stability_entropy(labels):
    labels = [l for l in labels if l is not None]
    if not labels:
        return None

    c = Counter(labels)
    total = sum(c.values())
    ent = 0.0
    for _, n in c.items():
        p = n / total
        ent -= p * math.log(p + 1e-12)
    return ent


# correlations
def pearson_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))

    if den_x == 0 or den_y == 0:
        return None

    return num / (den_x * den_y)


def average_ranks(values):
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])

    ranks = [0.0] * len(values)
    i = 0
    n = len(indexed)

    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank

        i = j + 1

    return ranks


def spearman_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    rx = average_ranks(xs)
    ry = average_ranks(ys)
    return pearson_corr(rx, ry)


# repeatability
def icc_oneway_average(prediction_matrix):
    if not prediction_matrix:
        return None

    n = len(prediction_matrix)
    k = len(prediction_matrix[0])

    if n < 2 or k < 2:
        return None

    if any(len(row) != k for row in prediction_matrix):
        return None

    row_means = [sum(row) / k for row in prediction_matrix]
    grand_mean = sum(row_means) / n

    ss_between = k * sum((rm - grand_mean) ** 2 for rm in row_means)

    ss_within = 0.0
    for row, rm in zip(prediction_matrix, row_means):
        ss_within += sum((x - rm) ** 2 for x in row)

    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))

    denom = ms_between + (k - 1) * ms_within
    if denom == 0:
        return None

    return (ms_between - ms_within) / denom


def build_icc_matrix_from_items(items):
    if not items:
        return []

    run_counts = [len(it["valid_preds"]) for it in items if len(it["valid_preds"]) >= 2]
    if not run_counts:
        return []

    modal_k = Counter(run_counts).most_common(1)[0][0]

    matrix = []
    for it in items:
        if len(it["valid_preds"]) == modal_k:
            try:
                row = [float(x) for x in it["valid_preds"]]
                matrix.append(row)
            except Exception:
                continue

    return matrix


# build the result tables
def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_JSONL}")

    # holistic scores
    detail_rows = []
    bucket = defaultdict(list)              # (model, condition) -> per-essay rows
    by_gold_bucket = defaultdict(list)      # (model, condition, gold) -> per-essay rows
    confusion_store = defaultdict(list)     # (model, condition) -> list of (gold, pred)

    # analytic scores
    dim_detail_rows = []
    dim_bucket = defaultdict(list)              # (model, condition, dimension) -> per-essay rows
    dim_by_gold_bucket = defaultdict(list)      # (model, condition, dimension, gold) -> per-essay rows
    dim_confusion_store = defaultdict(list)     # (model, condition, dimension) -> list of (gold, pred)

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            essay_id = obj.get("text_id_kaggle", f"line_{line_idx}")
            prompt_text = obj.get("prompt", None)

            # human scores

            gold_overall = clean_score(obj.get("Overall"))

            preds_block = obj.get("ellipse_result")
            if not isinstance(preds_block, dict) or not preds_block:
                continue

            for model_name, model_dict in preds_block.items():
                if not isinstance(model_dict, dict):
                    continue

                for condition_name, pred_list in model_dict.items():
                    if not isinstance(pred_list, list) or len(pred_list) == 0:
                        continue

                    # overall score
                    overall_pred_list = extract_prediction_list(pred_list, target_dimension="Overall")
                    overall_preds = [clean_score(x) for x in overall_pred_list]
                    valid_preds = [p for p in overall_preds if p is not None]

                    if gold_overall is not None and len(valid_preds) == EXPECTED_RUNS:
                        median_pred = median_score(valid_preds)
                        median_correct = (median_pred == gold_overall) if median_pred is not None else None
                        median_adjacent = adjacent_correct(median_pred, gold_overall) if median_pred is not None else None
                        median_mae = mean_absolute_error_single(median_pred, gold_overall) if median_pred is not None else None
                        median_signed_error = signed_error_single(median_pred, gold_overall) if median_pred is not None else None

                        mean_pred = mean_prediction(valid_preds)
                        pred_std = std_prediction(valid_preds)
                        mean_mae = abs(mean_pred - float(gold_overall)) if mean_pred is not None else None
                        mean_signed_error = (mean_pred - float(gold_overall)) if mean_pred is not None else None

                        correct_count = sum(1 for p in valid_preds if p == gold_overall)
                        correct_rate = correct_count / len(valid_preds)

                        adjacent_count = sum(1 for p in valid_preds if adjacent_correct(p, gold_overall))
                        adjacent_rate = adjacent_count / len(valid_preds)

                        unanimous = stability_unanimous(valid_preds)
                        ent = stability_entropy(valid_preds)

                        detail = {
                            "text_id_kaggle": essay_id,
                            "prompt": prompt_text,
                            "gold_overall": gold_overall,
                            "model": model_name,
                            "condition": condition_name,
                            "n_runs": len(valid_preds),
                            "predictions": ", ".join(valid_preds),
                            "valid_preds": valid_preds,

                            "median_score": median_pred,
                            "median_correct": median_correct,
                            "median_adjacent_correct": median_adjacent,
                            "median_mae": median_mae,
                            "median_signed_error": median_signed_error,

                            "mean_prediction": mean_pred,
                            "std_prediction": pred_std,
                            "mean_mae": mean_mae,
                            "mean_signed_error": mean_signed_error,

                            "correct_count": correct_count,
                            "correct_rate": correct_rate,
                            "adjacent_count": adjacent_count,
                            "adjacent_rate": adjacent_rate,

                            "unanimous": unanimous,
                            "entropy": ent,
                        }

                        detail_rows.append(detail)
                        bucket[(model_name, condition_name)].append(detail)
                        by_gold_bucket[(model_name, condition_name, gold_overall)].append(detail)

                        if median_pred is not None:
                            confusion_store[(model_name, condition_name)].append((gold_overall, median_pred))

                    if is_analytic_pred_list(pred_list):
                        # dimension scores
                        for dim in DIMENSIONS:
                            gold_dim = clean_score(obj.get(dim))
                            if gold_dim is None:
                                continue

                            dim_pred_list = extract_prediction_list(pred_list, target_dimension=dim)
                            dim_preds = [clean_score(x) for x in dim_pred_list]
                            dim_valid_preds = [p for p in dim_preds if p is not None]

                            if len(dim_valid_preds) != EXPECTED_RUNS:
                                continue

                            median_pred = median_score(dim_valid_preds)
                            median_correct = (median_pred == gold_dim) if median_pred is not None else None
                            median_adjacent = adjacent_correct(median_pred, gold_dim) if median_pred is not None else None
                            median_mae = mean_absolute_error_single(median_pred, gold_dim) if median_pred is not None else None
                            median_signed_error = signed_error_single(median_pred, gold_dim) if median_pred is not None else None

                            mean_pred = mean_prediction(dim_valid_preds)
                            pred_std = std_prediction(dim_valid_preds)
                            mean_mae = abs(mean_pred - float(gold_dim)) if mean_pred is not None else None
                            mean_signed_error = (mean_pred - float(gold_dim)) if mean_pred is not None else None

                            correct_count = sum(1 for p in dim_valid_preds if p == gold_dim)
                            correct_rate = correct_count / len(dim_valid_preds)

                            adjacent_count = sum(1 for p in dim_valid_preds if adjacent_correct(p, gold_dim))
                            adjacent_rate = adjacent_count / len(dim_valid_preds)

                            unanimous = stability_unanimous(dim_valid_preds)
                            ent = stability_entropy(dim_valid_preds)

                            dim_detail = {
                                "text_id_kaggle": essay_id,
                                "prompt": prompt_text,
                                "dimension": dim,
                                "gold_score": gold_dim,
                                "model": model_name,
                                "condition": condition_name,
                                "n_runs": len(dim_valid_preds),
                                "predictions": ", ".join(dim_valid_preds),
                                "valid_preds": dim_valid_preds,

                                "median_score": median_pred,
                                "median_correct": median_correct,
                                "median_adjacent_correct": median_adjacent,
                                "median_mae": median_mae,
                                "median_signed_error": median_signed_error,

                                "mean_prediction": mean_pred,
                                "std_prediction": pred_std,
                                "mean_mae": mean_mae,
                                "mean_signed_error": mean_signed_error,

                                "correct_count": correct_count,
                                "correct_rate": correct_rate,
                                "adjacent_count": adjacent_count,
                                "adjacent_rate": adjacent_rate,

                                "unanimous": unanimous,
                                "entropy": ent,
                            }

                            dim_detail_rows.append(dim_detail)
                            dim_bucket[(model_name, condition_name, dim)].append(dim_detail)
                            dim_by_gold_bucket[(model_name, condition_name, dim, gold_dim)].append(dim_detail)

                            if median_pred is not None:
                                dim_confusion_store[(model_name, condition_name, dim)].append((gold_dim, median_pred))

    # overall summary
    summary_rows = []

    for (model_name, condition_name), items in bucket.items():
        median_items = [it for it in items if it["median_score"] is not None]

        median_accuracy = None
        median_adjacent_accuracy = None
        qwk_median = None
        qwk_mean_rounded = None
        pearson_median = None
        spearman_median = None
        pearson_mean = None
        spearman_mean = None
        icc_runs = None

        if median_items:
            median_accuracy = sum(1 for it in median_items if it["median_correct"]) / len(median_items)
            median_adjacent_accuracy = sum(1 for it in median_items if it["median_adjacent_correct"]) / len(median_items)

            y_true = [SCORE_TO_INT[it["gold_overall"]] for it in median_items]
            y_pred_median = [SCORE_TO_INT[it["median_score"]] for it in median_items]
            qwk_median = quadratic_weighted_kappa(y_true, y_pred_median, n_classes=len(SCORE_ORDER))

            gold_vals = [float(it["gold_overall"]) for it in median_items]
            median_vals = [float(it["median_score"]) for it in median_items]
            pearson_median = pearson_corr(gold_vals, median_vals)
            spearman_median = spearman_corr(gold_vals, median_vals)

        mean_items = [it for it in items if it["mean_prediction"] is not None]
        if mean_items:
            rounded_mean_preds = []
            for it in mean_items:
                pred_val = it["mean_prediction"]
                nearest = min(SCORE_ORDER, key=lambda s: abs(float(s) - pred_val))
                rounded_mean_preds.append(SCORE_TO_INT[nearest])

            y_true_mean = [SCORE_TO_INT[it["gold_overall"]] for it in mean_items]
            qwk_mean_rounded = quadratic_weighted_kappa(y_true_mean, rounded_mean_preds, n_classes=len(SCORE_ORDER))

            gold_vals = [float(it["gold_overall"]) for it in mean_items]
            mean_vals = [it["mean_prediction"] for it in mean_items]
            pearson_mean = pearson_corr(gold_vals, mean_vals)
            spearman_mean = spearman_corr(gold_vals, mean_vals)

        icc_matrix = build_icc_matrix_from_items(items)
        if icc_matrix:
            icc_runs = icc_oneway_average(icc_matrix)

        avg_correct_count = sum(it["correct_count"] for it in items) / len(items) if items else None
        avg_correct_rate = sum(it["correct_rate"] for it in items) / len(items) if items else None
        avg_adjacent_rate = sum(it["adjacent_rate"] for it in items) / len(items) if items else None

        avg_median_mae = (
            sum(it["median_mae"] for it in items if it["median_mae"] is not None) / len(items)
            if items else None
        )
        avg_mean_mae = (
            sum(it["mean_mae"] for it in items if it["mean_mae"] is not None) / len(items)
            if items else None
        )

        avg_median_signed_error = (
            sum(it["median_signed_error"] for it in items if it["median_signed_error"] is not None) / len(items)
            if items else None
        )

        avg_mean_signed_error = (
            sum(it["mean_signed_error"] for it in items if it["mean_signed_error"] is not None) / len(items)
            if items else None
        )

        avg_std_prediction = (
            sum(it["std_prediction"] for it in items if it["std_prediction"] is not None) / len(items)
            if items else None
        )

        valid_unanimous = [it["unanimous"] for it in items if it["unanimous"] is not None]
        unanimous_rate = sum(1 for u in valid_unanimous if u) / len(valid_unanimous) if valid_unanimous else None

        valid_entropy = [it["entropy"] for it in items if it["entropy"] is not None]
        avg_entropy = sum(valid_entropy) / len(valid_entropy) if valid_entropy else None

        summary_rows.append({
            "model": model_name,
            "condition": condition_name,
            "valid_essays": len(items),
            "avg_runs_per_essay": (sum(it["n_runs"] for it in items) / len(items)) if items else None,

            "median_accuracy": median_accuracy,
            "median_adjacent_accuracy": median_adjacent_accuracy,
            "QWK_median": qwk_median,
            "QWK_mean_rounded": qwk_mean_rounded,

            "Pearson_median": pearson_median,
            "Spearman_median": spearman_median,
            "Pearson_mean": pearson_mean,
            "Spearman_mean": spearman_mean,
            "ICC_runs": icc_runs,

            "avg_correct_count_per_essay": avg_correct_count,
            "avg_correct_rate_per_essay": avg_correct_rate,
            "avg_adjacent_rate_per_essay": avg_adjacent_rate,

            "avg_median_MAE": avg_median_mae,
            "avg_mean_MAE": avg_mean_mae,

            "avg_median_signed_error": avg_median_signed_error,
            "avg_mean_signed_error": avg_mean_signed_error,

            "avg_std_prediction": avg_std_prediction,
            "unanimous_rate": unanimous_rate,
            "avg_entropy": avg_entropy,
        })

    # results by human score
    by_gold_rows = []

    for (model_name, condition_name, gold_score), items in by_gold_bucket.items():
        n_essays = len(items)

        median_accuracy = sum(1 for it in items if it["median_correct"]) / n_essays if n_essays else None
        adjacent_accuracy = sum(1 for it in items if it["median_adjacent_correct"]) / n_essays if n_essays else None

        mean_median_mae = (
            sum(it["median_mae"] for it in items if it["median_mae"] is not None) / n_essays
            if n_essays else None
        )

        mean_pred = (
            sum(it["mean_prediction"] for it in items if it["mean_prediction"] is not None) / n_essays
            if n_essays else None
        )

        mean_signed_error = (
            sum(it["mean_signed_error"] for it in items if it["mean_signed_error"] is not None) / n_essays
            if n_essays else None
        )

        mean_std = (
            sum(it["std_prediction"] for it in items if it["std_prediction"] is not None) / n_essays
            if n_essays else None
        )

        by_gold_rows.append({
            "model": model_name,
            "condition": condition_name,
            "gold_overall": gold_score,
            "n_essays": n_essays,
            "median_accuracy": median_accuracy,
            "median_adjacent_accuracy": adjacent_accuracy,
            "avg_median_MAE": mean_median_mae,
            "avg_mean_prediction": mean_pred,
            "avg_mean_signed_error": mean_signed_error,
            "avg_std_prediction": mean_std,
        })

    # overall confusion table
    confusion_rows = []

    for (model_name, condition_name), pairs in confusion_store.items():
        matrix = {gold: {pred: 0 for pred in SCORE_ORDER} for gold in SCORE_ORDER}

        for gold, pred in pairs:
            if gold in matrix and pred in matrix[gold]:
                matrix[gold][pred] += 1

        for gold in SCORE_ORDER:
            row = {
                "model": model_name,
                "condition": condition_name,
                "gold_overall": gold,
            }
            for pred in SCORE_ORDER:
                row[f"pred_{pred}"] = matrix[gold][pred]
            confusion_rows.append(row)

    # dimension summary
    dim_summary_rows = []

    for (model_name, condition_name, dim), items in dim_bucket.items():
        median_items = [it for it in items if it["median_score"] is not None]

        median_accuracy = None
        median_adjacent_accuracy = None
        qwk_median = None
        qwk_mean_rounded = None
        pearson_median = None
        spearman_median = None
        pearson_mean = None
        spearman_mean = None
        icc_runs = None

        if median_items:
            median_accuracy = sum(1 for it in median_items if it["median_correct"]) / len(median_items)
            median_adjacent_accuracy = sum(1 for it in median_items if it["median_adjacent_correct"]) / len(median_items)

            y_true = [SCORE_TO_INT[it["gold_score"]] for it in median_items]
            y_pred_median = [SCORE_TO_INT[it["median_score"]] for it in median_items]
            qwk_median = quadratic_weighted_kappa(y_true, y_pred_median, n_classes=len(SCORE_ORDER))

            gold_vals = [float(it["gold_score"]) for it in median_items]
            median_vals = [float(it["median_score"]) for it in median_items]
            pearson_median = pearson_corr(gold_vals, median_vals)
            spearman_median = spearman_corr(gold_vals, median_vals)

        mean_items = [it for it in items if it["mean_prediction"] is not None]
        if mean_items:
            rounded_mean_preds = []
            for it in mean_items:
                pred_val = it["mean_prediction"]
                nearest = min(SCORE_ORDER, key=lambda s: abs(float(s) - pred_val))
                rounded_mean_preds.append(SCORE_TO_INT[nearest])

            y_true_mean = [SCORE_TO_INT[it["gold_score"]] for it in mean_items]
            qwk_mean_rounded = quadratic_weighted_kappa(y_true_mean, rounded_mean_preds, n_classes=len(SCORE_ORDER))

            gold_vals = [float(it["gold_score"]) for it in mean_items]
            mean_vals = [it["mean_prediction"] for it in mean_items]
            pearson_mean = pearson_corr(gold_vals, mean_vals)
            spearman_mean = spearman_corr(gold_vals, mean_vals)

        icc_matrix = build_icc_matrix_from_items(items)
        if icc_matrix:
            icc_runs = icc_oneway_average(icc_matrix)

        avg_correct_count = sum(it["correct_count"] for it in items) / len(items) if items else None
        avg_correct_rate = sum(it["correct_rate"] for it in items) / len(items) if items else None
        avg_adjacent_rate = sum(it["adjacent_rate"] for it in items) / len(items) if items else None

        avg_median_mae = (
            sum(it["median_mae"] for it in items if it["median_mae"] is not None) / len(items)
            if items else None
        )
        avg_mean_mae = (
            sum(it["mean_mae"] for it in items if it["mean_mae"] is not None) / len(items)
            if items else None
        )

        avg_median_signed_error = (
            sum(it["median_signed_error"] for it in items if it["median_signed_error"] is not None) / len(items)
            if items else None
        )

        avg_mean_signed_error = (
            sum(it["mean_signed_error"] for it in items if it["mean_signed_error"] is not None) / len(items)
            if items else None
        )

        avg_std_prediction = (
            sum(it["std_prediction"] for it in items if it["std_prediction"] is not None) / len(items)
            if items else None
        )

        valid_unanimous = [it["unanimous"] for it in items if it["unanimous"] is not None]
        unanimous_rate = sum(1 for u in valid_unanimous if u) / len(valid_unanimous) if valid_unanimous else None

        valid_entropy = [it["entropy"] for it in items if it["entropy"] is not None]
        avg_entropy = sum(valid_entropy) / len(valid_entropy) if valid_entropy else None

        dim_summary_rows.append({
            "model": model_name,
            "condition": condition_name,
            "dimension": dim,
            "valid_essays": len(items),
            "avg_runs_per_essay": (sum(it["n_runs"] for it in items) / len(items)) if items else None,

            "median_accuracy": median_accuracy,
            "median_adjacent_accuracy": median_adjacent_accuracy,
            "QWK_median": qwk_median,
            "QWK_mean_rounded": qwk_mean_rounded,

            "Pearson_median": pearson_median,
            "Spearman_median": spearman_median,
            "Pearson_mean": pearson_mean,
            "Spearman_mean": spearman_mean,
            "ICC_runs": icc_runs,

            "avg_correct_count_per_essay": avg_correct_count,
            "avg_correct_rate_per_essay": avg_correct_rate,
            "avg_adjacent_rate_per_essay": avg_adjacent_rate,

            "avg_median_MAE": avg_median_mae,
            "avg_mean_MAE": avg_mean_mae,

            "avg_median_signed_error": avg_median_signed_error,
            "avg_mean_signed_error": avg_mean_signed_error,

            "avg_std_prediction": avg_std_prediction,
            "unanimous_rate": unanimous_rate,
            "avg_entropy": avg_entropy,
        })

    # dimensions by human score
    dim_by_gold_rows = []

    for (model_name, condition_name, dim, gold_score), items in dim_by_gold_bucket.items():
        n_essays = len(items)

        median_accuracy = sum(1 for it in items if it["median_correct"]) / n_essays if n_essays else None
        adjacent_accuracy = sum(1 for it in items if it["median_adjacent_correct"]) / n_essays if n_essays else None

        mean_median_mae = (
            sum(it["median_mae"] for it in items if it["median_mae"] is not None) / n_essays
            if n_essays else None
        )

        mean_pred = (
            sum(it["mean_prediction"] for it in items if it["mean_prediction"] is not None) / n_essays
            if n_essays else None
        )

        mean_signed_error = (
            sum(it["mean_signed_error"] for it in items if it["mean_signed_error"] is not None) / n_essays
            if n_essays else None
        )

        mean_std = (
            sum(it["std_prediction"] for it in items if it["std_prediction"] is not None) / n_essays
            if n_essays else None
        )

        dim_by_gold_rows.append({
            "model": model_name,
            "condition": condition_name,
            "dimension": dim,
            "gold_score": gold_score,
            "n_essays": n_essays,
            "median_accuracy": median_accuracy,
            "median_adjacent_accuracy": adjacent_accuracy,
            "avg_median_MAE": mean_median_mae,
            "avg_mean_prediction": mean_pred,
            "avg_mean_signed_error": mean_signed_error,
            "avg_std_prediction": mean_std,
        })

    # dimension confusion tables
    dim_confusion_rows = []

    for (model_name, condition_name, dim), pairs in dim_confusion_store.items():
        matrix = {gold: {pred: 0 for pred in SCORE_ORDER} for gold in SCORE_ORDER}

        for gold, pred in pairs:
            if gold in matrix and pred in matrix[gold]:
                matrix[gold][pred] += 1

        for gold in SCORE_ORDER:
            row = {
                "model": model_name,
                "condition": condition_name,
                "dimension": dim,
                "gold_score": gold,
            }
            for pred in SCORE_ORDER:
                row[f"pred_{pred}"] = matrix[gold][pred]
            dim_confusion_rows.append(row)

    # final tables
    df_summary = pd.DataFrame(summary_rows)
    if not df_summary.empty:
        df_summary = df_summary.sort_values(["model", "condition"]).reset_index(drop=True)

    df_detail = pd.DataFrame(detail_rows)
    if not df_detail.empty:
        df_detail = df_detail.sort_values(["model", "condition", "text_id_kaggle"]).reset_index(drop=True)

    df_by_gold = pd.DataFrame(by_gold_rows)
    if not df_by_gold.empty:
        df_by_gold = df_by_gold.sort_values(["model", "condition", "gold_overall"]).reset_index(drop=True)

    df_confusion = pd.DataFrame(confusion_rows)
    if not df_confusion.empty:
        df_confusion = df_confusion.sort_values(["model", "condition", "gold_overall"]).reset_index(drop=True)

    df_dim_summary = pd.DataFrame(dim_summary_rows)
    if not df_dim_summary.empty:
        df_dim_summary = df_dim_summary.sort_values(["model", "condition", "dimension"]).reset_index(drop=True)

    df_dim_detail = pd.DataFrame(dim_detail_rows)
    if not df_dim_detail.empty:
        df_dim_detail = df_dim_detail.sort_values(["model", "condition", "dimension", "text_id_kaggle"]).reset_index(drop=True)

    df_dim_by_gold = pd.DataFrame(dim_by_gold_rows)
    if not df_dim_by_gold.empty:
        df_dim_by_gold = df_dim_by_gold.sort_values(["model", "condition", "dimension", "gold_score"]).reset_index(drop=True)

    df_dim_confusion = pd.DataFrame(dim_confusion_rows)
    if not df_dim_confusion.empty:
        df_dim_confusion = df_dim_confusion.sort_values(["model", "condition", "dimension", "gold_score"]).reset_index(drop=True)

    # round values for export
    round_cols_summary = [
        "avg_runs_per_essay",
        "median_accuracy",
        "median_adjacent_accuracy",
        "QWK_median",
        "QWK_mean_rounded",
        "Pearson_median",
        "Spearman_median",
        "Pearson_mean",
        "Spearman_mean",
        "ICC_runs",
        "avg_correct_count_per_essay",
        "avg_correct_rate_per_essay",
        "avg_adjacent_rate_per_essay",
        "avg_median_MAE",
        "avg_mean_MAE",
        "avg_median_signed_error",
        "avg_mean_signed_error",
        "avg_std_prediction",
        "unanimous_rate",
        "avg_entropy",
    ]
    for col in round_cols_summary:
        if col in df_summary.columns:
            df_summary[col] = pd.to_numeric(df_summary[col], errors="coerce").round(4)
        if col in df_dim_summary.columns:
            df_dim_summary[col] = pd.to_numeric(df_dim_summary[col], errors="coerce").round(4)

    round_cols_detail = [
        "median_mae",
        "median_signed_error",
        "mean_prediction",
        "std_prediction",
        "mean_mae",
        "mean_signed_error",
        "correct_rate",
        "adjacent_rate",
        "entropy",
    ]
    for col in round_cols_detail:
        if col in df_detail.columns:
            df_detail[col] = pd.to_numeric(df_detail[col], errors="coerce").round(4)
        if col in df_dim_detail.columns:
            df_dim_detail[col] = pd.to_numeric(df_dim_detail[col], errors="coerce").round(4)

    round_cols_by_gold = [
        "median_accuracy",
        "median_adjacent_accuracy",
        "avg_median_MAE",
        "avg_mean_prediction",
        "avg_mean_signed_error",
        "avg_std_prediction",
    ]
    for col in round_cols_by_gold:
        if col in df_by_gold.columns:
            df_by_gold[col] = pd.to_numeric(df_by_gold[col], errors="coerce").round(4)
        if col in df_dim_by_gold.columns:
            df_dim_by_gold[col] = pd.to_numeric(df_dim_by_gold[col], errors="coerce").round(4)

    # write the workbook
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="Summary")
        df_detail.to_excel(writer, index=False, sheet_name="PerEssay")
        df_by_gold.to_excel(writer, index=False, sheet_name="ByGoldScore")
        df_confusion.to_excel(writer, index=False, sheet_name="ConfusionMedian")

        df_dim_summary.to_excel(writer, index=False, sheet_name="DimensionSummary")
        df_dim_detail.to_excel(writer, index=False, sheet_name="DimensionPerEssay")
        df_dim_by_gold.to_excel(writer, index=False, sheet_name="DimensionByGold")
        df_dim_confusion.to_excel(writer, index=False, sheet_name="DimensionConfusionMedian")

    print(f"Wrote {len(df_summary)} result summaries to {OUTPUT_XLSX}.")


if __name__ == "__main__":
    main()
