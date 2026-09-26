"""Paired tests for the three rubric conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from test_baseline import (
    clean_score,
    holm_adjust,
    percentile_interval,
    quadratic_weighted_kappa,
)
from paths import ELLIPSE_RESULTS, RESULTS_DIR


MODELS = [
    ("GPT-5.5", "openai::gpt-5.5"),
    ("Gemini 3", "gemini::gemini-3-flash-preview"),
    ("DeepSeek-V3.2", "deepseek::deepseek-chat"),
    ("GPT-5.4", "openai::gpt-5.4"),
    ("DeepSeek-V4", "deepseek::deepseek-v4-flash"),
    ("Gemini 3.1 Lite", "gemini::gemini-3.1-flash-lite-preview"),
    ("GPT-5.4 Nano", "openai::gpt-5.4-nano"),
]

ANALYTIC_MODELS = {
    "GPT-5.5",
    "Gemini 3",
    "DeepSeek-V3.2",
    "GPT-5.4",
    "DeepSeek-V4",
}

CONDITIONS = {
    "N": "essay_no_rubric",
    "H": "essay_only",
    "A": "essay_plus_analytic_rubric",
}

COMPARISONS = [("H", "N"), ("A", "H"), ("A", "N")]
METRICS = ["QWK", "MAE"]


def models_for_comparison(comparison: tuple[str, str]) -> set[str]:
    return {name for name, _ in MODELS} if comparison == ("H", "N") else ANALYTIC_MODELS


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
        default=RESULTS_DIR / "statistical_tests" / "rubric_conditions",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_runs(values: object, condition: str) -> np.ndarray:
    if not isinstance(values, list) or len(values) != 5:
        count = 0 if values is None else len(values)
        raise ValueError(f"Expected five scores for {condition}, found {count}")

    if condition == "A":
        scores = [item["Overall"] for item in values]
    else:
        scores = values
    return np.asarray([clean_score(value) for value in scores], dtype=float)


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # array order is essay, model, condition, run
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    human = np.asarray([clean_score(record["Overall"]) for record in records])
    runs = np.full(
        (len(records), len(MODELS), len(CONDITIONS), 5), np.nan, dtype=float
    )
    condition_names = list(CONDITIONS)

    for essay_index, record in enumerate(records):
        results = record["ellipse_result"]
        for model_index, (model_name, model_key) in enumerate(MODELS):
            model_results = results.get(model_key, {})
            for condition_index, short_name in enumerate(condition_names):
                full_name = CONDITIONS[short_name]
                if short_name == "A" and model_name not in ANALYTIC_MODELS:
                    continue
                try:
                    runs[essay_index, model_index, condition_index] = read_runs(
                        model_results.get(full_name), short_name
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid data for essay {essay_index + 1}, "
                        f"{model_name}, condition {short_name}"
                    ) from exc

    return human, runs


def metric_value(metric: str, human: np.ndarray, prediction: np.ndarray) -> float:
    if metric == "QWK":
        return quadratic_weighted_kappa(human, prediction)
    if metric == "MAE":
        return float(np.mean(np.abs(prediction - human)))
    raise ValueError(f"Unsupported metric: {metric}")


def permutation_p_value(
    human: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    metric: str,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    observed = metric_value(metric, human, prediction_a) - metric_value(
        metric, human, prediction_b
    )
    extreme = 0

    for _ in range(permutations):
        swap = rng.random(len(human)) < 0.5
        permuted_a = np.where(swap, prediction_b, prediction_a)
        permuted_b = np.where(swap, prediction_a, prediction_b)
        difference = metric_value(metric, human, permuted_a) - metric_value(
            metric, human, permuted_b
        )
        extreme += abs(difference) >= abs(observed)

    return (extreme + 1) / (permutations + 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human, runs = load_data(args.input)
    # collapse the five runs before comparing prompt conditions
    medians = np.median(runs, axis=3)
    condition_index = {name: index for index, name in enumerate(CONDITIONS)}
    strata = [np.flatnonzero(human == score) for score in np.unique(human)]
    rng = np.random.default_rng(args.seed)

    boot_diff = {}
    for comparison in COMPARISONS:
        for model_index, (model_name, _) in enumerate(MODELS):
            if model_name not in models_for_comparison(comparison):
                continue
            for metric in METRICS:
                boot_diff[comparison, model_name, metric] = np.empty(
                    args.bootstrap, dtype=float
                )

    for iteration in range(args.bootstrap):
        # both prompt conditions use the same sampled essays
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        )
        sampled_human = human[sampled]

        for condition_a, condition_b in COMPARISONS:
            index_a = condition_index[condition_a]
            index_b = condition_index[condition_b]
            for model_index, (model_name, _) in enumerate(MODELS):
                if model_name not in models_for_comparison((condition_a, condition_b)):
                    continue
                prediction_a = medians[sampled, model_index, index_a]
                prediction_b = medians[sampled, model_index, index_b]
                for metric in METRICS:
                    boot_diff[
                        (condition_a, condition_b), model_name, metric
                    ][iteration] = metric_value(
                        metric, sampled_human, prediction_a
                    ) - metric_value(metric, sampled_human, prediction_b)

    permutation_rng = np.random.default_rng(args.seed + 1)
    rows = []
    for condition_a, condition_b in COMPARISONS:
        index_a = condition_index[condition_a]
        index_b = condition_index[condition_b]
        for model_index, (model_name, _) in enumerate(MODELS):
            if model_name not in models_for_comparison((condition_a, condition_b)):
                continue
            prediction_a = medians[:, model_index, index_a]
            prediction_b = medians[:, model_index, index_b]
            for metric in METRICS:
                estimate_a = metric_value(metric, human, prediction_a)
                estimate_b = metric_value(metric, human, prediction_b)
                differences = boot_diff[
                    (condition_a, condition_b), model_name, metric
                ]
                ci_low, ci_high = percentile_interval(differences)
                rows.append(
                    {
                        "model": model_name,
                        "comparison": f"{condition_a} - {condition_b}",
                        "metric": metric,
                        "estimate_a": estimate_a,
                        "estimate_b": estimate_b,
                        "difference": estimate_a - estimate_b,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "permutation_p": permutation_p_value(
                            human,
                            prediction_a,
                            prediction_b,
                            metric,
                            args.permutations,
                            permutation_rng,
                        ),
                    }
                )

    # adjust each planned comparison and metric separately
    for comparison in ["H - N", "A - H", "A - N"]:
        for metric in METRICS:
            family = [
                row
                for row in rows
                if row["comparison"] == comparison and row["metric"] == metric
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
            "sampling_unit": "essay",
            "stratification": "human Overall score",
            "aggregation": "median of five repeated runs",
            "multiple_testing": (
                "Holm correction across five models within each comparison and metric"
            ),
        },
        "results": rows,
    }

    json_path = args.output_dir / "rubric_condition_bootstrap_results.json"
    csv_path = args.output_dir / "rubric_condition_comparisons.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
