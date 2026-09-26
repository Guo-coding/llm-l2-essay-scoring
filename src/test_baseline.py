"""Bootstrap intervals and model comparisons for the baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from paths import ELLIPSE_RESULTS, RESULTS_DIR, display_path


MODELS = [
    ("GPT-5.5", "openai::gpt-5.5", "main"),
    ("Gemini 3", "gemini::gemini-3-flash-preview", "main"),
    ("DeepSeek-V3.2", "deepseek::deepseek-chat", "main"),
    ("Gemini 3.1 Lite", "gemini::gemini-3.1-flash-lite-preview", "lightweight"),
    ("GPT-5.4", "openai::gpt-5.4", "main"),
    ("DeepSeek-V4", "deepseek::deepseek-v4-flash", "main"),
    ("GPT-5.4 nano", "openai::gpt-5.4-nano", "lightweight"),
]

KEY_COMPARISONS = [
    ("GPT-5.5", "Gemini 3"),
    ("GPT-5.5", "DeepSeek-V3.2"),
    ("Gemini 3", "DeepSeek-V3.2"),
    ("GPT-5.5", "GPT-5.4"),
    ("DeepSeek-V3.2", "DeepSeek-V4"),
    ("Gemini 3", "Gemini 3.1 Lite"),
    ("GPT-5.4", "GPT-5.4 nano"),
]

SCORE_MIN = 1.0
SCORE_MAX = 5.0
SCORE_STEP = 0.5
N_SCORE_LEVELS = 9
CONDITION = "essay_only"

METRICS = [
    "Accuracy",
    "Adjacent Accuracy",
    "QWK",
    "MAE",
    "MSD",
    "Spearman",
    "ICC",
    "SD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ELLIPSE_RESULTS,
        help="JSONL file containing the five-run scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "statistical_tests" / "baseline",
        help="Directory for JSON and CSV results.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_score(value: object) -> float:
    score = float(value)
    doubled = round(score * 2)
    if not np.isclose(score * 2, doubled):
        raise ValueError(f"Score is not in 0.5-point steps: {value!r}")
    if not SCORE_MIN <= score <= SCORE_MAX:
        raise ValueError(f"Score is outside 1.0-5.0: {value!r}")
    return score


def median_score(scores: np.ndarray) -> float:
    if len(scores) != 5:
        raise ValueError(f"Expected five repeated scores, found {len(scores)}")
    return float(np.median(scores))


def load_baseline(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    # keep human scores and five model runs in aligned arrays
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc

    n_essays = len(records)
    n_models = len(MODELS)
    human = np.empty(n_essays, dtype=float)
    runs = np.empty((n_essays, n_models, 5), dtype=float)
    qc_rows = []

    for essay_index, record in enumerate(records):
        human[essay_index] = clean_score(record["Overall"])
        result_block = record.get("ellipse_result", {})

        for model_index, (model_name, model_key, model_group) in enumerate(MODELS):
            values = result_block.get(model_key, {}).get(CONDITION)
            if not isinstance(values, list) or len(values) != 5:
                raise ValueError(
                    f"{model_name} has {0 if values is None else len(values)} runs "
                    f"for essay {essay_index + 1}; expected 5."
                )
            runs[essay_index, model_index] = [clean_score(value) for value in values]

    score_counts = Counter(human)
    for model_index, (model_name, model_key, model_group) in enumerate(MODELS):
        model_runs = runs[:, model_index, :]
        qc_rows.append(
            {
                "model": model_name,
                "model_key": model_key,
                "group": model_group,
                "essays": n_essays,
                "runs_per_essay": 5,
                "valid_scores": int(model_runs.size),
                "minimum_score": float(model_runs.min()),
                "maximum_score": float(model_runs.max()),
            }
        )

    qc_summary = {
        "input_file": display_path(path),
        "condition": CONDITION,
        "essays": n_essays,
        "models": n_models,
        "runs_per_essay": 5,
        "human_score_counts": {
            f"{score:.1f}": int(score_counts[score]) for score in sorted(score_counts)
        },
    }
    return human, runs, qc_rows, qc_summary


def score_classes(scores: np.ndarray) -> np.ndarray:
    return np.rint((scores - SCORE_MIN) / SCORE_STEP).astype(int)


def quadratic_weighted_kappa(human: np.ndarray, prediction: np.ndarray) -> float:
    human_class = score_classes(human)
    prediction_class = score_classes(prediction)

    observed = np.zeros((N_SCORE_LEVELS, N_SCORE_LEVELS), dtype=float)
    np.add.at(observed, (human_class, prediction_class), 1)

    human_counts = np.bincount(human_class, minlength=N_SCORE_LEVELS)
    prediction_counts = np.bincount(prediction_class, minlength=N_SCORE_LEVELS)
    expected = np.outer(human_counts, prediction_counts) / len(human)

    positions = np.arange(N_SCORE_LEVELS)
    weights = ((positions[:, None] - positions[None, :]) / (N_SCORE_LEVELS - 1)) ** 2
    denominator = float(np.sum(weights * expected))
    if denominator == 0:
        return float("nan")
    return 1.0 - float(np.sum(weights * observed)) / denominator


def spearman_correlation(human: np.ndarray, prediction: np.ndarray) -> float:
    human_rank = rankdata(human, method="average")
    prediction_rank = rankdata(prediction, method="average")
    return float(np.corrcoef(human_rank, prediction_rank)[0, 1])


def icc_oneway_single(runs: np.ndarray) -> float:
    """ICC(1,1): one-way random effects, single measurement."""
    n_essays, n_runs = runs.shape
    essay_means = runs.mean(axis=1)
    grand_mean = essay_means.mean()

    between_ss = n_runs * np.sum((essay_means - grand_mean) ** 2)
    within_ss = np.sum((runs - essay_means[:, None]) ** 2)
    between_ms = between_ss / (n_essays - 1)
    within_ms = within_ss / (n_essays * (n_runs - 1))
    denominator = between_ms + (n_runs - 1) * within_ms
    if denominator == 0:
        return float("nan")
    return float((between_ms - within_ms) / denominator)


def calculate_metrics(
    human: np.ndarray,
    median: np.ndarray,
    runs: np.ndarray,
) -> dict[str, float]:
    error = median - human
    return {
        "Accuracy": float(np.mean(error == 0)),
        "Adjacent Accuracy": float(np.mean(np.abs(error) <= 0.5)),
        "QWK": quadratic_weighted_kappa(human, median),
        "MAE": float(np.mean(np.abs(error))),
        "MSD": float(np.mean(error)),
        "Spearman": spearman_correlation(human, median),
        "ICC": icc_oneway_single(runs),
        "SD": float(np.mean(np.std(runs, axis=1, ddof=0))),
    }


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def favored_model(metric: str, model_a: str, model_b: str, difference: float) -> str:
    if metric in {"MAE", "SD"}:
        return model_a if difference < 0 else model_b
    if metric in {"MSD"}:
        return "See signed difference"
    return model_a if difference > 0 else model_b


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in their original order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        corrected = (len(p_values) - rank) * p_values[index]
        running_max = max(running_max, corrected)
        adjusted[index] = min(running_max, 1.0)
    return adjusted.tolist()


def paired_permutation_p_value(
    human: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    metric: str,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided paired randomization test using essay-level score swaps."""
    if metric == "QWK":
        observed = quadratic_weighted_kappa(human, prediction_a) - quadratic_weighted_kappa(
            human, prediction_b
        )
    elif metric == "MAE":
        observed = np.mean(np.abs(prediction_a - human)) - np.mean(
            np.abs(prediction_b - human)
        )
    else:
        raise ValueError(f"Unsupported permutation metric: {metric}")

    extreme = 0
    for _ in range(permutations):
        swap = rng.random(len(human)) < 0.5
        permuted_a = np.where(swap, prediction_b, prediction_a)
        permuted_b = np.where(swap, prediction_a, prediction_b)
        if metric == "QWK":
            difference = quadratic_weighted_kappa(
                human, permuted_a
            ) - quadratic_weighted_kappa(human, permuted_b)
        else:
            difference = np.mean(np.abs(permuted_a - human)) - np.mean(
                np.abs(permuted_b - human)
            )
        extreme += abs(difference) >= abs(observed)

    return (extreme + 1) / (permutations + 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human, runs, qc_rows, qc_summary = load_baseline(args.input)
    # all accuracy and error measures use the median run
    median = np.apply_along_axis(median_score, 2, runs)
    n_essays, n_models = median.shape

    point_estimates = []
    point_by_model = {}
    for model_index, (model_name, model_key, model_group) in enumerate(MODELS):
        metrics = calculate_metrics(human, median[:, model_index], runs[:, model_index, :])
        point_by_model[model_name] = metrics
        point_estimates.append(
            {
                "model": model_name,
                "model_key": model_key,
                "group": model_group,
                **metrics,
            }
        )

    strata = [np.flatnonzero(human == score) for score in np.unique(human)]
    rng = np.random.default_rng(args.seed)
    bootstrap_values = {
        metric: np.empty((args.bootstrap, n_models), dtype=float) for metric in METRICS
    }

    # sample within score levels and reuse the essays for every model
    for iteration in range(args.bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        )
        sampled_human = human[sampled]

        for model_index in range(n_models):
            metrics = calculate_metrics(
                sampled_human,
                median[sampled, model_index],
                runs[sampled, model_index, :],
            )
            for metric in METRICS:
                bootstrap_values[metric][iteration, model_index] = metrics[metric]

    model_ci_rows = []
    for model_index, (model_name, model_key, model_group) in enumerate(MODELS):
        row = {
            "model": model_name,
            "model_key": model_key,
            "group": model_group,
            "essays": n_essays,
        }
        for metric in METRICS:
            lower, upper = percentile_interval(bootstrap_values[metric][:, model_index])
            row[f"{metric}_estimate"] = point_by_model[model_name][metric]
            row[f"{metric}_ci_low"] = lower
            row[f"{metric}_ci_high"] = upper
        row["MSD_below_zero_95"] = row["MSD_ci_high"] < 0
        model_ci_rows.append(row)

    model_index_by_name = {name: index for index, (name, _, _) in enumerate(MODELS)}
    all_pairs = []
    for first_index in range(n_models):
        for second_index in range(first_index + 1, n_models):
            first_name = MODELS[first_index][0]
            second_name = MODELS[second_index][0]
            for metric in ["QWK", "MAE"]:
                differences = (
                    bootstrap_values[metric][:, first_index]
                    - bootstrap_values[metric][:, second_index]
                )
                lower, upper = percentile_interval(differences)
                point_difference = (
                    point_by_model[first_name][metric] - point_by_model[second_name][metric]
                )
                supported = lower > 0 or upper < 0
                all_pairs.append(
                    {
                        "model_a": first_name,
                        "model_b": second_name,
                        "comparison": f"{first_name} - {second_name}",
                        "metric": metric,
                        "difference_a_minus_b": point_difference,
                        "ci_low": lower,
                        "ci_high": upper,
                        "supported_95": supported,
                        "favored_model": (
                            favored_model(metric, first_name, second_name, point_difference)
                            if supported
                            else "No clear difference"
                        ),
                    }
                )

    key_pair_names = set(KEY_COMPARISONS)
    key_pairs = [
        row
        for row in all_pairs
        if (row["model_a"], row["model_b"]) in key_pair_names
        or (row["model_b"], row["model_a"]) in key_pair_names
    ]

    permutation_rng = np.random.default_rng(args.seed + 1)

    # permutation tests are limited to the planned comparisons
    for row in key_pairs:
        first_index = model_index_by_name[row["model_a"]]
        second_index = model_index_by_name[row["model_b"]]
        row["permutation_p"] = paired_permutation_p_value(
            human,
            median[:, first_index],
            median[:, second_index],
            row["metric"],
            args.permutations,
            permutation_rng,
        )

    for metric in ["QWK", "MAE"]:
        # one Holm family for each tested metric
        metric_rows = [row for row in key_pairs if row["metric"] == metric]
        adjusted = holm_adjust([row["permutation_p"] for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["holm_adjusted_p"] = adjusted_p
            row["significant_holm_05"] = adjusted_p < 0.05

    result = {
        "method": {
            "name": "Stratified paired percentile bootstrap",
            "bootstrap_samples": args.bootstrap,
            "permutation_samples": args.permutations,
            "confidence_level": 0.95,
            "seed": args.seed,
            "sampling_unit": "essay",
            "stratification": "human Overall score",
            "aggregation": "median of five repeated runs",
            "pairing": "the same sampled essay indices were used for every model",
            "multiple_testing": "Holm correction within QWK and MAE across seven planned comparisons",
        },
        "quality_control": qc_summary,
        "quality_control_by_model": qc_rows,
        "model_confidence_intervals": model_ci_rows,
        "key_pairwise_comparisons": key_pairs,
        "all_pairwise_comparisons": all_pairs,
    }

    result_path = args.output_dir / "baseline_bootstrap_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output_dir / "baseline_model_confidence_intervals.csv", model_ci_rows)
    write_csv(args.output_dir / "baseline_key_comparisons.csv", key_pairs)
    write_csv(args.output_dir / "baseline_all_pairwise_comparisons.csv", all_pairs)
    write_csv(args.output_dir / "baseline_data_quality.csv", qc_rows)

    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
