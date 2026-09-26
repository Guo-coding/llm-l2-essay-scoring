#!/usr/bin/env python3
"""Paired tests for the feature-group ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from test_baseline import (
    holm_adjust,
    paired_permutation_p_value,
    percentile_interval,
    quadratic_weighted_kappa,
)
from run_fusion_ablation import (
    FEATURE_GROUPS,
    MODELS,
    N_SPLITS,
    RANDOM_STATE,
    load_rows,
    ridge_model,
    round_to_half,
)
from paths import RESULTS_DIR


METRICS = ("QWK", "MAE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "statistical_tests" / "ablation",
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


def out_of_fold_predictions(
    features: pd.DataFrame,
    human: np.ndarray,
    strata: np.ndarray,
) -> np.ndarray:
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    # each row is predicted only in its held-out fold
    predictions = np.full(len(features), np.nan)
    for train_index, test_index in splitter.split(features, strata):
        model = ridge_model()
        model.fit(features.iloc[train_index], human[train_index])
        predictions[test_index] = round_to_half(model.predict(features.iloc[test_index]))
    if np.isnan(predictions).any():
        raise ValueError("Out-of-fold prediction array contains missing values")
    return predictions


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_comparisons() -> tuple[list[dict], list[dict]]:
    data, feature_data = load_rows()
    metric_columns = feature_data["feature_name"].tolist()
    group_columns = {
        group_label: feature_data.loc[
            feature_data["feature_group"] == group_name,
            "feature_name",
        ].tolist()
        for group_label, group_name in FEATURE_GROUPS
    }
    if any(not columns for columns in group_columns.values()):
        raise ValueError("At least one linguistic feature group is empty")

    comparisons = []
    prediction_rows = []

    for model_name, model_key in MODELS:
        model_data = data.loc[data["model_key"] == model_key].copy().reset_index(drop=True)
        if len(model_data) != 1200 or model_data["essay_id"].duplicated().any():
            raise ValueError(f"Unexpected essay rows for {model_name}: {len(model_data)}")

        human = model_data["gold"].to_numpy(dtype=float)
        strata = model_data["gold"].astype(str).to_numpy()
        calibrated = out_of_fold_predictions(
            model_data[["llm_median"]].astype(float),
            human,
            strata,
        )
        full_fusion = out_of_fold_predictions(
            model_data[["llm_median"] + metric_columns].astype(float),
            human,
            strata,
        )

        # add one group to the calibrated score
        for group_label, _ in FEATURE_GROUPS:
            added_columns = group_columns[group_label]
            add_one = out_of_fold_predictions(
                model_data[["llm_median"] + added_columns].astype(float),
                human,
                strata,
            )
            comparisons.append(
                {
                    "design": "Add-one",
                    "model": model_name,
                    "model_key": model_key,
                    "feature_group": group_label,
                    "setting_a": f"LLM + {group_label}",
                    "setting_b": "Calibrated LLM",
                    "human": human,
                    "prediction_a": add_one,
                    "prediction_b": calibrated,
                }
            )
            for index in range(len(model_data)):
                prediction_rows.append(
                    {
                        "essay_id": model_data.loc[index, "essay_id"],
                        "model": model_name,
                        "design": "Add-one",
                        "feature_group": group_label,
                        "gold": human[index],
                        "reference_pred": calibrated[index],
                        "ablation_pred": add_one[index],
                    }
                )

        # remove one group from the full fusion model
        for group_label, _ in FEATURE_GROUPS:
            removed = set(group_columns[group_label])
            remaining = [column for column in metric_columns if column not in removed]
            leave_one = out_of_fold_predictions(
                model_data[["llm_median"] + remaining].astype(float),
                human,
                strata,
            )
            comparisons.append(
                {
                    "design": "Leave-one-out",
                    "model": model_name,
                    "model_key": model_key,
                    "feature_group": group_label,
                    "setting_a": f"All except {group_label}",
                    "setting_b": "Full fusion",
                    "human": human,
                    "prediction_a": leave_one,
                    "prediction_b": full_fusion,
                }
            )
            for index in range(len(model_data)):
                prediction_rows.append(
                    {
                        "essay_id": model_data.loc[index, "essay_id"],
                        "model": model_name,
                        "design": "Leave-one-out",
                        "feature_group": group_label,
                        "gold": human[index],
                        "reference_pred": full_fusion[index],
                        "ablation_pred": leave_one[index],
                    }
                )

    return comparisons, prediction_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons, prediction_rows = build_comparisons()

    bootstrap_rng = np.random.default_rng(args.seed)
    boot_diff = {
        (index, metric): np.empty(args.bootstrap, dtype=float)
        for index in range(len(comparisons))
        for metric in METRICS
    }
    human = comparisons[0]["human"]
    strata = [np.flatnonzero(human == score) for score in np.unique(human)]

    # use the same bootstrap sample for each feature-group comparison
    for iteration in range(args.bootstrap):
        sampled = np.concatenate(
            [bootstrap_rng.choice(group, size=len(group), replace=True) for group in strata]
        )
        for index, comparison in enumerate(comparisons):
            sampled_human = comparison["human"][sampled]
            prediction_a = comparison["prediction_a"][sampled]
            prediction_b = comparison["prediction_b"][sampled]
            for metric in METRICS:
                boot_diff[index, metric][iteration] = metric_value(
                    metric, sampled_human, prediction_a
                ) - metric_value(metric, sampled_human, prediction_b)
    permutation_rng = np.random.default_rng(args.seed + 1)
    rows = []
    for index, comparison in enumerate(comparisons):
        human = comparison["human"]
        prediction_a = comparison["prediction_a"]
        prediction_b = comparison["prediction_b"]
        for metric in METRICS:
            estimate_a = metric_value(metric, human, prediction_a)
            estimate_b = metric_value(metric, human, prediction_b)
            differences = boot_diff[index, metric]
            ci_low, ci_high = percentile_interval(differences)
            rows.append(
                {
                    "design": comparison["design"],
                    "model": comparison["model"],
                    "feature_group": comparison["feature_group"],
                    "setting_a": comparison["setting_a"],
                    "setting_b": comparison["setting_b"],
                    "metric": metric,
                    "estimate_a": estimate_a,
                    "estimate_b": estimate_b,
                    "difference": estimate_a - estimate_b,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "permutation_p": paired_permutation_p_value(
                        human,
                        prediction_a,
                        prediction_b,
                        metric,
                        args.permutations,
                        permutation_rng,
                    ),
                }
            )

    for design in ("Add-one", "Leave-one-out"):
        for metric in METRICS:
            family = [
                row
                for row in rows
                if row["design"] == design and row["metric"] == metric
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
                "Holm correction across 20 comparisons within each ablation "
                "design and metric"
            ),
        },
        "results": rows,
    }

    write_csv(args.output_dir / "ablation_comparisons.csv", rows)
    write_csv(args.output_dir / "ablation_predictions.csv", prediction_rows)
    (args.output_dir / "ablation_bootstrap_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
