#!/usr/bin/env python3
"""Paired tests for linguistic features added to the prompt."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from test_baseline import (
    calculate_metrics,
    clean_score,
    holm_adjust,
    paired_permutation_p_value,
    percentile_interval,
    quadratic_weighted_kappa,
)
from paths import ELLIPSE_RESULTS, RESULTS_DIR


MODELS = [
    ("GPT-5.4", "openai::gpt-5.4"),
    ("Gemini 3", "gemini::gemini-3-flash-preview"),
    ("DeepSeek-V4", "deepseek::deepseek-v4-flash"),
]

CONDITIONS = {
    "Holistic": "essay_only",
    "Holistic + features": "essay_plus_all_metrics",
}

METRICS = ["QWK", "MAE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ELLIPSE_RESULTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "statistical_tests" / "linguistic_prompt",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_runs(values: object, model: str, condition: str) -> np.ndarray:
    if not isinstance(values, list) or len(values) != 5:
        count = 0 if values is None else len(values)
        raise ValueError(
            f"Expected five scores for {model}, {condition}; found {count}"
        )
    return np.asarray([clean_score(value) for value in values], dtype=float)


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # keep the two prompt conditions in the same essay order
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    human = np.asarray([clean_score(record["Overall"]) for record in records])
    runs = np.empty((len(records), len(MODELS), len(CONDITIONS), 5), dtype=float)

    for essay_index, record in enumerate(records):
        result_block = record.get("ellipse_result", {})
        for model_index, (model_name, model_key) in enumerate(MODELS):
            model_results = result_block.get(model_key, {})
            for condition_index, (condition_name, condition_key) in enumerate(
                CONDITIONS.items()
            ):
                runs[essay_index, model_index, condition_index] = read_runs(
                    model_results.get(condition_key), model_name, condition_name
                )

    return human, runs


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human, runs = load_data(args.input)
    # prompt comparisons use the median of the five runs
    medians = np.median(runs, axis=3)
    condition_names = list(CONDITIONS)
    holistic_index = condition_names.index("Holistic")
    feature_index = condition_names.index("Holistic + features")

    estimate_rows = []
    for model_index, (model_name, _) in enumerate(MODELS):
        for condition_index, condition_name in enumerate(condition_names):
            values = calculate_metrics(
                human,
                medians[:, model_index, condition_index],
                runs[:, model_index, condition_index, :],
            )
            estimate_rows.append(
                {
                    "model": model_name,
                    "condition": condition_name,
                    **values,
                }
            )

    strata = [np.flatnonzero(human == score) for score in np.unique(human)]
    bootstrap_rng = np.random.default_rng(args.seed)
    boot_diff = {
        (model_name, metric): np.empty(args.bootstrap, dtype=float)
        for model_name, _ in MODELS
        for metric in METRICS
    }

    # keep the two prompt conditions paired by essay
    for iteration in range(args.bootstrap):
        sampled = np.concatenate(
            [bootstrap_rng.choice(group, size=len(group), replace=True) for group in strata]
        )
        sampled_human = human[sampled]

        for model_index, (model_name, _) in enumerate(MODELS):
            feature_prediction = medians[sampled, model_index, feature_index]
            holistic_prediction = medians[sampled, model_index, holistic_index]
            for metric in METRICS:
                boot_diff[model_name, metric][iteration] = metric_value(
                    metric, sampled_human, feature_prediction
                ) - metric_value(metric, sampled_human, holistic_prediction)

    permutation_rng = np.random.default_rng(args.seed + 1)
    comparison_rows = []
    for model_index, (model_name, _) in enumerate(MODELS):
        feature_prediction = medians[:, model_index, feature_index]
        holistic_prediction = medians[:, model_index, holistic_index]
        for metric in METRICS:
            feature_estimate = metric_value(metric, human, feature_prediction)
            holistic_estimate = metric_value(metric, human, holistic_prediction)
            differences = boot_diff[model_name, metric]
            ci_low, ci_high = percentile_interval(differences)
            comparison_rows.append(
                {
                    "model": model_name,
                    "comparison": "Holistic + features - Holistic",
                    "metric": metric,
                    "feature_estimate": feature_estimate,
                    "holistic_estimate": holistic_estimate,
                    "difference": feature_estimate - holistic_estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "permutation_p": paired_permutation_p_value(
                        human,
                        feature_prediction,
                        holistic_prediction,
                        metric,
                        args.permutations,
                        permutation_rng,
                    ),
                }
            )

    for metric in METRICS:
        family = [row for row in comparison_rows if row["metric"] == metric]
        adjusted = holm_adjust([row["permutation_p"] for row in family])
        for row, adjusted_p in zip(family, adjusted):
            row["holm_adjusted_p"] = adjusted_p
            row["significant_holm_05"] = adjusted_p < 0.05

    result = {
        "method": {
            "bootstrap_samples": args.bootstrap,
            "permutation_samples": args.permutations,
            "confidence_level": 0.95,
            "sampling_unit": "essay",
            "stratification": "human Overall score",
            "aggregation": "median of five repeated runs",
            "multiple_testing": (
                f"Holm correction across {len(MODELS)} models within each metric"
            ),
        },
        "estimates": estimate_rows,
        "comparisons": comparison_rows,
    }

    write_csv(args.output_dir / "linguistic_prompt_estimates.csv", estimate_rows)
    write_csv(args.output_dir / "linguistic_prompt_comparisons.csv", comparison_rows)
    (args.output_dir / "linguistic_prompt_bootstrap_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
