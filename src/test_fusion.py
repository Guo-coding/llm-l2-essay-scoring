#!/usr/bin/env python3
"""Paired tests for calibration and feature fusion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from test_baseline import (
    holm_adjust,
    paired_permutation_p_value,
    percentile_interval,
    quadratic_weighted_kappa,
)
from paths import RESULTS_DIR


MODELS = ["GPT-5.5", "Gemini 3", "DeepSeek-V3.2", "GPT-5.4", "DeepSeek-V4"]

COMPARISONS = [
    ("Calibrated LLM - Raw LLM", "calibrated_pred", "raw_median_pred"),
    ("Fusion - Raw LLM", "fusion_pred", "raw_median_pred"),
    ("Fusion - Calibrated LLM", "fusion_pred", "calibrated_pred"),
]

METRICS = ["QWK", "MAE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "fusion" / "csv" / "predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "statistical_tests" / "fusion",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metric_value(metric: str, human: np.ndarray, prediction: np.ndarray) -> float:
    if metric == "QWK":
        return quadratic_weighted_kappa(human, prediction)
    if metric == "MAE":
        return float(np.mean(np.abs(prediction - human)))
    raise ValueError(f"Unsupported metric: {metric}")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path: Path) -> dict[str, pd.DataFrame]:
    # these are the held-out predictions written by the fusion script
    data = pd.read_csv(path)
    required = {
        "essay_id",
        "model",
        "gold",
        "raw_median_pred",
        "calibrated_pred",
        "fusion_pred",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    model_data = {}
    for model in MODELS:
        subset = data.loc[data["model"] == model].copy()
        subset = subset.sort_values("essay_id").reset_index(drop=True)
        if len(subset) != 1200 or subset["essay_id"].duplicated().any():
            raise ValueError(f"Unexpected prediction rows for {model}: {len(subset)}")
        if subset[list(required - {"essay_id", "model"})].isna().any().any():
            raise ValueError(f"Missing prediction values for {model}")
        model_data[model] = subset

    # every model must refer to the same essays in the same order
    reference_ids = model_data[MODELS[0]]["essay_id"].tolist()
    reference_gold = model_data[MODELS[0]]["gold"].to_numpy(dtype=float)
    for model in MODELS[1:]:
        if model_data[model]["essay_id"].tolist() != reference_ids:
            raise ValueError(f"Essay ordering differs for {model}")
        if not np.array_equal(
            model_data[model]["gold"].to_numpy(dtype=float), reference_gold
        ):
            raise ValueError(f"Human scores differ for {model}")

    return model_data


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_data = load_predictions(args.input)

    bootstrap_rng = np.random.default_rng(args.seed)
    boot_diff = {
        (comparison_name, model, metric): np.empty(args.bootstrap, dtype=float)
        for comparison_name, _, _ in COMPARISONS
        for model in MODELS
        for metric in METRICS
    }

    human = model_data[MODELS[0]]["gold"].to_numpy(dtype=float)
    strata = [np.flatnonzero(human == score) for score in np.unique(human)]

    # keep the two settings paired during resampling
    for iteration in range(args.bootstrap):
        sampled = np.concatenate(
            [bootstrap_rng.choice(group, size=len(group), replace=True) for group in strata]
        )
        sampled_human = human[sampled]

        for model in MODELS:
            frame = model_data[model]
            for comparison_name, column_a, column_b in COMPARISONS:
                prediction_a = frame[column_a].to_numpy(dtype=float)[sampled]
                prediction_b = frame[column_b].to_numpy(dtype=float)[sampled]
                for metric in METRICS:
                    boot_diff[
                        comparison_name, model, metric
                    ][iteration] = metric_value(
                        metric, sampled_human, prediction_a
                    ) - metric_value(metric, sampled_human, prediction_b)

    permutation_rng = np.random.default_rng(args.seed + 1)
    rows = []
    for comparison_name, column_a, column_b in COMPARISONS:
        for model in MODELS:
            frame = model_data[model]
            model_human = frame["gold"].to_numpy(dtype=float)
            prediction_a = frame[column_a].to_numpy(dtype=float)
            prediction_b = frame[column_b].to_numpy(dtype=float)
            for metric in METRICS:
                estimate_a = metric_value(metric, model_human, prediction_a)
                estimate_b = metric_value(metric, model_human, prediction_b)
                differences = boot_diff[comparison_name, model, metric]
                ci_low, ci_high = percentile_interval(differences)
                rows.append(
                    {
                        "model": model,
                        "comparison": comparison_name,
                        "metric": metric,
                        "estimate_a": estimate_a,
                        "estimate_b": estimate_b,
                        "difference": estimate_a - estimate_b,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "permutation_p": paired_permutation_p_value(
                            model_human,
                            prediction_a,
                            prediction_b,
                            metric,
                            args.permutations,
                            permutation_rng,
                        ),
                    }
                )

    for comparison_name, _, _ in COMPARISONS:
        for metric in METRICS:
            # correction is applied across the five models
            family = [
                row
                for row in rows
                if row["comparison"] == comparison_name and row["metric"] == metric
            ]
            adjusted = holm_adjust([row["permutation_p"] for row in family])
            for row, adjusted_p in zip(family, adjusted):
                row["holm_adjusted_p"] = adjusted_p
                row["significant_holm_05"] = adjusted_p < 0.05

    result = {
        "method": {
            "bootstrap_samples": args.bootstrap,
            "permutation_samples": args.permutations,
            "confidence_level": 0.95,
            "sampling_unit": "essay-level out-of-fold prediction",
            "stratification": "human Overall score",
            "multiple_testing": (
                "Holm correction across five models within each comparison and metric"
            ),
        },
        "results": rows,
    }

    write_csv(args.output_dir / "fusion_comparisons.csv", rows)
    (args.output_dir / "fusion_bootstrap_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
