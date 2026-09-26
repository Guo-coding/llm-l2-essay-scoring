"""Tests for the six analytic dimensions."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from test_baseline import (
    clean_score,
    holm_adjust,
    icc_oneway_single,
    percentile_interval,
    quadratic_weighted_kappa,
    spearman_correlation,
)
from paths import ELLIPSE_RESULTS, RESULTS_DIR


MODELS = [
    ("GPT-5.5", "openai::gpt-5.5"),
    ("Gemini 3", "gemini::gemini-3-flash-preview"),
    ("DeepSeek-V3.2", "deepseek::deepseek-chat"),
    ("GPT-5.4", "openai::gpt-5.4"),
    ("DeepSeek-V4", "deepseek::deepseek-v4-flash"),
]

DIMENSIONS = [
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
]

TARGETS = ["Overall", *DIMENSIONS]
CONDITION = "essay_plus_analytic_rubric"
SCORE_MIN = 1.0
SCORE_STEP = 0.5
N_SCORE_LEVELS = 9
METRICS = ["QWK", "MAE"]

MODEL_COMPARISONS = [
    ("GPT-5.5", "Gemini 3"),
    ("GPT-5.5", "DeepSeek-V3.2"),
    ("GPT-5.5", "GPT-5.4"),
    ("GPT-5.5", "DeepSeek-V4"),
    ("Gemini 3", "DeepSeek-V3.2"),
    ("Gemini 3", "GPT-5.4"),
    ("Gemini 3", "DeepSeek-V4"),
    ("DeepSeek-V3.2", "GPT-5.4"),
    ("DeepSeek-V3.2", "DeepSeek-V4"),
    ("GPT-5.4", "DeepSeek-V4"),
]


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
        default=RESULTS_DIR / "statistical_tests" / "analytic_dimensions",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--permutation-chunk", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
    return records


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # dimensions stay in the rubric order used throughout the thesis
    records = read_records(path)
    human = np.empty((len(records), len(TARGETS)), dtype=float)
    runs = np.empty((len(records), len(MODELS), len(TARGETS), 5), dtype=float)

    for essay_index, record in enumerate(records):
        for target_index, target in enumerate(TARGETS):
            human[essay_index, target_index] = clean_score(record[target])

        results = record.get("ellipse_result", {})
        for model_index, (model_name, model_key) in enumerate(MODELS):
            values = results.get(model_key, {}).get(CONDITION)
            if not isinstance(values, list) or len(values) != 5:
                count = 0 if values is None else len(values)
                raise ValueError(
                    f"{model_name}, essay {essay_index + 1}: expected five "
                    f"analytic runs, found {count}"
                )
            for run_index, item in enumerate(values):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{model_name}, essay {essay_index + 1}, run "
                        f"{run_index + 1}: analytic result is not an object"
                    )
                for target_index, target in enumerate(TARGETS):
                    runs[essay_index, model_index, target_index, run_index] = (
                        clean_score(item[target])
                    )

    return human, runs


def score_classes(scores: np.ndarray) -> np.ndarray:
    return np.rint((scores - SCORE_MIN) / SCORE_STEP).astype(np.int8)


def qwk_batch(human_class: np.ndarray, prediction_class: np.ndarray) -> np.ndarray:
    """Calculate QWK for each row of two score-class matrices."""
    prediction_class = np.asarray(prediction_class, dtype=np.int16)
    if prediction_class.ndim == 1:
        prediction_class = prediction_class[None, :]

    human_class = np.asarray(human_class, dtype=np.int16)
    if human_class.ndim == 1:
        human_class = np.broadcast_to(human_class, prediction_class.shape)
    if human_class.shape != prediction_class.shape:
        raise ValueError("Human and prediction matrices must have the same shape")

    batch_size, n_essays = prediction_class.shape
    row = np.repeat(np.arange(batch_size, dtype=np.int64), n_essays)

    observed_code = (
        row * (N_SCORE_LEVELS**2)
        + (human_class.ravel() * N_SCORE_LEVELS + prediction_class.ravel())
    )
    observed = np.bincount(
        observed_code, minlength=batch_size * N_SCORE_LEVELS**2
    ).reshape(batch_size, N_SCORE_LEVELS, N_SCORE_LEVELS)

    human_code = row * N_SCORE_LEVELS + human_class.ravel()
    prediction_code = row * N_SCORE_LEVELS + prediction_class.ravel()
    human_counts = np.bincount(
        human_code, minlength=batch_size * N_SCORE_LEVELS
    ).reshape(batch_size, N_SCORE_LEVELS)
    prediction_counts = np.bincount(
        prediction_code, minlength=batch_size * N_SCORE_LEVELS
    ).reshape(batch_size, N_SCORE_LEVELS)

    positions = np.arange(N_SCORE_LEVELS)
    weights = ((positions[:, None] - positions[None, :]) / (N_SCORE_LEVELS - 1)) ** 2
    numerator = np.einsum("bij,ij->b", observed, weights)
    denominator = np.einsum(
        "bi,ij,bj->b", human_counts, weights, prediction_counts
    ) / n_essays
    return 1.0 - numerator / denominator


def cell_metrics(
    human_dimensions: np.ndarray,
    median_dimensions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model-by-dimension matrices for QWK and MAE."""
    n_models = median_dimensions.shape[1]
    n_dimensions = human_dimensions.shape[1]
    qwk = np.empty((n_models, n_dimensions), dtype=float)
    mae = np.empty((n_models, n_dimensions), dtype=float)

    for model_index in range(n_models):
        for dimension_index in range(n_dimensions):
            human = human_dimensions[:, dimension_index]
            prediction = median_dimensions[:, model_index, dimension_index]
            qwk[model_index, dimension_index] = quadratic_weighted_kappa(
                human, prediction
            )
            mae[model_index, dimension_index] = np.mean(np.abs(prediction - human))
    return qwk, mae


def full_cell_metrics(
    human_dimensions: np.ndarray,
    median_dimensions: np.ndarray,
    dimension_runs: np.ndarray,
) -> dict[str, np.ndarray]:
    n_models = median_dimensions.shape[1]
    n_dimensions = human_dimensions.shape[1]
    values = {
        metric: np.empty((n_models, n_dimensions), dtype=float)
        for metric in [
            "Accuracy",
            "Adjacent Accuracy",
            "QWK",
            "MAE",
            "MSD",
            "Spearman",
            "ICC",
        ]
    }

    for model_index in range(n_models):
        for dimension_index in range(n_dimensions):
            human = human_dimensions[:, dimension_index]
            prediction = median_dimensions[:, model_index, dimension_index]
            error = prediction - human
            values["Accuracy"][model_index, dimension_index] = np.mean(error == 0)
            values["Adjacent Accuracy"][model_index, dimension_index] = np.mean(
                np.abs(error) <= 0.5
            )
            values["QWK"][model_index, dimension_index] = quadratic_weighted_kappa(
                human, prediction
            )
            values["MAE"][model_index, dimension_index] = np.mean(np.abs(error))
            values["MSD"][model_index, dimension_index] = np.mean(error)
            values["Spearman"][model_index, dimension_index] = spearman_correlation(
                human, prediction
            )
            values["ICC"][model_index, dimension_index] = icc_oneway_single(
                dimension_runs[:, model_index, dimension_index, :]
            )
    return values


def model_pair_permutation(
    human: np.ndarray,
    predictions: np.ndarray,
    first_index: int,
    second_index: int,
    permutations: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    pred_a = predictions[:, first_index, :]
    pred_b = predictions[:, second_index, :]
    human_class = score_classes(human)
    pred_a_class = score_classes(pred_a)
    pred_b_class = score_classes(pred_b)

    qwk_matrix, mae_matrix = cell_metrics(
        human, predictions[:, [first_index, second_index], :]
    )
    observed_qwk = float(qwk_matrix[0].mean() - qwk_matrix[1].mean())
    observed_mae = float(mae_matrix[0].mean() - mae_matrix[1].mean())
    extreme_qwk = 0
    extreme_mae = 0
    completed = 0

    while completed < permutations:
        size = min(chunk_size, permutations - completed)
        swap = rng.random((size, len(human))) < 0.5
        swap_3d = swap[:, :, None]

        perm_a = np.where(swap_3d, pred_b[None, :, :], pred_a[None, :, :])
        perm_b = np.where(swap_3d, pred_a[None, :, :], pred_b[None, :, :])
        diff_mae = np.mean(np.abs(perm_a - human[None, :, :]), axis=(1, 2)) - np.mean(
            np.abs(perm_b - human[None, :, :]), axis=(1, 2)
        )

        perm_a_class = np.where(
            swap_3d, pred_b_class[None, :, :], pred_a_class[None, :, :]
        )
        perm_b_class = np.where(
            swap_3d, pred_a_class[None, :, :], pred_b_class[None, :, :]
        )
        qwk_a = np.zeros(size, dtype=float)
        qwk_b = np.zeros(size, dtype=float)
        for dimension_index in range(len(DIMENSIONS)):
            qwk_a += qwk_batch(
                human_class[:, dimension_index], perm_a_class[:, :, dimension_index]
            )
            qwk_b += qwk_batch(
                human_class[:, dimension_index], perm_b_class[:, :, dimension_index]
            )
        diff_qwk = (qwk_a - qwk_b) / len(DIMENSIONS)

        extreme_qwk += int(np.sum(np.abs(diff_qwk) >= abs(observed_qwk)))
        extreme_mae += int(np.sum(np.abs(diff_mae) >= abs(observed_mae)))
        completed += size

    return {
        "QWK": (extreme_qwk + 1) / (permutations + 1),
        "MAE": (extreme_mae + 1) / (permutations + 1),
    }


def dimension_pair_permutation(
    human: np.ndarray,
    predictions: np.ndarray,
    first_index: int,
    second_index: int,
    permutations: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    human_a = human[:, first_index]
    human_b = human[:, second_index]
    pred_a = predictions[:, :, first_index]
    pred_b = predictions[:, :, second_index]
    human_a_class = score_classes(human_a)
    human_b_class = score_classes(human_b)
    pred_a_class = score_classes(pred_a)
    pred_b_class = score_classes(pred_b)

    qwk_matrix, mae_matrix = cell_metrics(human, predictions)
    observed_qwk = float(qwk_matrix[:, first_index].mean() - qwk_matrix[:, second_index].mean())
    observed_mae = float(mae_matrix[:, first_index].mean() - mae_matrix[:, second_index].mean())
    extreme_qwk = 0
    extreme_mae = 0
    completed = 0

    while completed < permutations:
        size = min(chunk_size, permutations - completed)
        swap = rng.random((size, len(human))) < 0.5
        swap_3d = swap[:, :, None]

        perm_human_a = np.where(swap, human_b[None, :], human_a[None, :])
        perm_human_b = np.where(swap, human_a[None, :], human_b[None, :])
        perm_a = np.where(swap_3d, pred_b[None, :, :], pred_a[None, :, :])
        perm_b = np.where(swap_3d, pred_a[None, :, :], pred_b[None, :, :])
        diff_mae = np.mean(
            np.abs(perm_a - perm_human_a[:, :, None]), axis=(1, 2)
        ) - np.mean(np.abs(perm_b - perm_human_b[:, :, None]), axis=(1, 2))

        perm_human_a_class = np.where(
            swap, human_b_class[None, :], human_a_class[None, :]
        )
        perm_human_b_class = np.where(
            swap, human_a_class[None, :], human_b_class[None, :]
        )
        perm_a_class = np.where(
            swap_3d, pred_b_class[None, :, :], pred_a_class[None, :, :]
        )
        perm_b_class = np.where(
            swap_3d, pred_a_class[None, :, :], pred_b_class[None, :, :]
        )
        qwk_a = np.zeros(size, dtype=float)
        qwk_b = np.zeros(size, dtype=float)
        for model_index in range(len(MODELS)):
            qwk_a += qwk_batch(
                perm_human_a_class, perm_a_class[:, :, model_index]
            )
            qwk_b += qwk_batch(
                perm_human_b_class, perm_b_class[:, :, model_index]
            )
        diff_qwk = (qwk_a - qwk_b) / len(MODELS)

        extreme_qwk += int(np.sum(np.abs(diff_qwk) >= abs(observed_qwk)))
        extreme_mae += int(np.sum(np.abs(diff_mae) >= abs(observed_mae)))
        completed += size

    return {
        "QWK": (extreme_qwk + 1) / (permutations + 1),
        "MAE": (extreme_mae + 1) / (permutations + 1),
    }


def overall_dimension_permutation(
    human_overall: np.ndarray,
    human_dimensions: np.ndarray,
    prediction_overall: np.ndarray,
    prediction_dimensions: np.ndarray,
    permutations: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    overall_qwk = quadratic_weighted_kappa(human_overall, prediction_overall)
    dimension_qwk = np.mean(
        [
            quadratic_weighted_kappa(
                human_dimensions[:, index], prediction_dimensions[:, index]
            )
            for index in range(len(DIMENSIONS))
        ]
    )
    observed_qwk = float(overall_qwk - dimension_qwk)
    observed_mae = float(
        np.mean(np.abs(prediction_overall - human_overall))
        - np.mean(np.abs(prediction_dimensions - human_dimensions))
    )

    overall_human_class = score_classes(human_overall)
    dimension_human_class = score_classes(human_dimensions)
    overall_prediction_class = score_classes(prediction_overall)
    dimension_prediction_class = score_classes(prediction_dimensions)
    extreme_qwk = 0
    extreme_mae = 0
    completed = 0

    while completed < permutations:
        size = min(chunk_size, permutations - completed)
        swap = rng.random((size, len(human_overall))) < 0.5
        first_qwk = np.zeros(size, dtype=float)
        second_qwk = np.zeros(size, dtype=float)
        first_mae = np.zeros(size, dtype=float)
        second_mae = np.zeros(size, dtype=float)

        for dimension_index in range(len(DIMENSIONS)):
            human_dimension = human_dimensions[:, dimension_index]
            prediction_dimension = prediction_dimensions[:, dimension_index]
            first_human = np.where(
                swap, human_dimension[None, :], human_overall[None, :]
            )
            second_human = np.where(
                swap, human_overall[None, :], human_dimension[None, :]
            )
            first_prediction = np.where(
                swap, prediction_dimension[None, :], prediction_overall[None, :]
            )
            second_prediction = np.where(
                swap, prediction_overall[None, :], prediction_dimension[None, :]
            )
            first_mae += np.mean(np.abs(first_prediction - first_human), axis=1)
            second_mae += np.mean(np.abs(second_prediction - second_human), axis=1)

            first_human_class = np.where(
                swap,
                dimension_human_class[None, :, dimension_index],
                overall_human_class[None, :],
            )
            second_human_class = np.where(
                swap,
                overall_human_class[None, :],
                dimension_human_class[None, :, dimension_index],
            )
            first_prediction_class = np.where(
                swap,
                dimension_prediction_class[None, :, dimension_index],
                overall_prediction_class[None, :],
            )
            second_prediction_class = np.where(
                swap,
                overall_prediction_class[None, :],
                dimension_prediction_class[None, :, dimension_index],
            )
            first_qwk += qwk_batch(first_human_class, first_prediction_class)
            second_qwk += qwk_batch(second_human_class, second_prediction_class)

        diff_qwk = (first_qwk - second_qwk) / len(DIMENSIONS)
        diff_mae = (first_mae - second_mae) / len(DIMENSIONS)
        extreme_qwk += int(np.sum(np.abs(diff_qwk) >= abs(observed_qwk)))
        extreme_mae += int(np.sum(np.abs(diff_mae) >= abs(observed_mae)))
        completed += size

    return {
        "QWK": (extreme_qwk + 1) / (permutations + 1),
        "MAE": (extreme_mae + 1) / (permutations + 1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_holm(rows: list[dict], family_field: str = "metric") -> None:
    for metric in METRICS:
        family = [row for row in rows if row[family_field] == metric]
        adjusted = holm_adjust([row["permutation_p"] for row in family])
        for row, adjusted_p in zip(family, adjusted):
            row["holm_adjusted_p"] = adjusted_p
            row["significant_holm_05"] = adjusted_p < 0.05


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human, runs = load_data(args.input)
    median = np.median(runs, axis=3)
    human_overall = human[:, 0]
    human_dimensions = human[:, 1:]
    median_overall = median[:, :, 0]
    median_dimensions = median[:, :, 1:]
    dimension_runs = runs[:, :, 1:, :]

    full_cells = full_cell_metrics(
        human_dimensions, median_dimensions, dimension_runs
    )
    model_rows = []
    for model_index, (model_name, model_key) in enumerate(MODELS):
        model_rows.append(
            {
                "model": model_name,
                "model_key": model_key,
                **{
                    metric: float(values[model_index].mean())
                    for metric, values in full_cells.items()
                },
            }
        )

    dimension_rows = []
    for dimension_index, dimension in enumerate(DIMENSIONS):
        dimension_rows.append(
            {
                "dimension": dimension,
                **{
                    metric: float(values[:, dimension_index].mean())
                    for metric, values in full_cells.items()
                },
            }
        )

    point_qwk, point_mae = cell_metrics(human_dimensions, median_dimensions)
    # macro averages give each dimension the same weight
    point_model = {"QWK": point_qwk.mean(axis=1), "MAE": point_mae.mean(axis=1)}
    point_dimension = {"QWK": point_qwk.mean(axis=0), "MAE": point_mae.mean(axis=0)}
    point_overall = {
        "QWK": np.asarray(
            [
                quadratic_weighted_kappa(human_overall, median_overall[:, index])
                for index in range(len(MODELS))
            ]
        ),
        "MAE": np.mean(np.abs(median_overall - human_overall[:, None]), axis=0),
    }

    strata = [np.flatnonzero(human_overall == score) for score in np.unique(human_overall)]
    bootstrap_rng = np.random.default_rng(args.seed)
    model_bootstrap = {
        metric: np.empty((args.bootstrap, len(MODELS)), dtype=float)
        for metric in METRICS
    }
    dimension_bootstrap = {
        metric: np.empty((args.bootstrap, len(DIMENSIONS)), dtype=float)
        for metric in METRICS
    }
    overall_difference_bootstrap = {
        metric: np.empty((args.bootstrap, len(MODELS)), dtype=float)
        for metric in METRICS
    }

    # reuse each bootstrap sample across models and dimensions
    for iteration in range(args.bootstrap):
        sampled = np.concatenate(
            [
                bootstrap_rng.choice(indices, size=len(indices), replace=True)
                for indices in strata
            ]
        )
        sampled_qwk, sampled_mae = cell_metrics(
            human_dimensions[sampled], median_dimensions[sampled]
        )
        model_bootstrap["QWK"][iteration] = sampled_qwk.mean(axis=1)
        model_bootstrap["MAE"][iteration] = sampled_mae.mean(axis=1)
        dimension_bootstrap["QWK"][iteration] = sampled_qwk.mean(axis=0)
        dimension_bootstrap["MAE"][iteration] = sampled_mae.mean(axis=0)

        for model_index in range(len(MODELS)):
            overall_qwk = quadratic_weighted_kappa(
                human_overall[sampled], median_overall[sampled, model_index]
            )
            overall_mae = np.mean(
                np.abs(
                    median_overall[sampled, model_index] - human_overall[sampled]
                )
            )
            overall_difference_bootstrap["QWK"][iteration, model_index] = (
                overall_qwk - model_bootstrap["QWK"][iteration, model_index]
            )
            overall_difference_bootstrap["MAE"][iteration, model_index] = (
                overall_mae - model_bootstrap["MAE"][iteration, model_index]
            )

    model_index = {name: index for index, (name, _) in enumerate(MODELS)}
    dimension_index = {name: index for index, name in enumerate(DIMENSIONS)}
    permutation_rng = np.random.default_rng(args.seed + 1)

    model_comparison_rows = []

    # compare models using the average across all six dimensions
    for first_name, second_name in MODEL_COMPARISONS:
        first = model_index[first_name]
        second = model_index[second_name]
        p_values = model_pair_permutation(
            human_dimensions,
            median_dimensions,
            first,
            second,
            args.permutations,
            args.permutation_chunk,
            permutation_rng,
        )
        for metric in METRICS:
            differences = (
                model_bootstrap[metric][:, first]
                - model_bootstrap[metric][:, second]
            )
            ci_low, ci_high = percentile_interval(differences)
            model_comparison_rows.append(
                {
                    "model_a": first_name,
                    "model_b": second_name,
                    "metric": metric,
                    "estimate_a": point_model[metric][first],
                    "estimate_b": point_model[metric][second],
                    "difference_a_minus_b": (
                        point_model[metric][first] - point_model[metric][second]
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "permutation_p": p_values[metric],
                }
            )
    add_holm(model_comparison_rows)

    dimension_comparison_rows = []
    # all dimension pairs are tested because no pair was chosen in advance
    for first_name, second_name in combinations(DIMENSIONS, 2):
        first = dimension_index[first_name]
        second = dimension_index[second_name]
        p_values = dimension_pair_permutation(
            human_dimensions,
            median_dimensions,
            first,
            second,
            args.permutations,
            args.permutation_chunk,
            permutation_rng,
        )
        for metric in METRICS:
            differences = (
                dimension_bootstrap[metric][:, first]
                - dimension_bootstrap[metric][:, second]
            )
            ci_low, ci_high = percentile_interval(differences)
            dimension_comparison_rows.append(
                {
                    "dimension_a": first_name,
                    "dimension_b": second_name,
                    "metric": metric,
                    "estimate_a": point_dimension[metric][first],
                    "estimate_b": point_dimension[metric][second],
                    "difference_a_minus_b": (
                        point_dimension[metric][first]
                        - point_dimension[metric][second]
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "permutation_p": p_values[metric],
                }
            )
    add_holm(dimension_comparison_rows)

    overall_comparison_rows = []
    for model_idx, (model_name, _) in enumerate(MODELS):
        p_values = overall_dimension_permutation(
            human_overall,
            human_dimensions,
            median_overall[:, model_idx],
            median_dimensions[:, model_idx, :],
            args.permutations,
            args.permutation_chunk,
            permutation_rng,
        )
        for metric in METRICS:
            differences = overall_difference_bootstrap[metric][
                :, model_idx
            ]
            ci_low, ci_high = percentile_interval(differences)
            overall_comparison_rows.append(
                {
                    "model": model_name,
                    "comparison": "Overall - six-dimension macro average",
                    "metric": metric,
                    "overall_estimate": point_overall[metric][model_idx],
                    "dimension_macro_estimate": point_model[metric][model_idx],
                    "difference": (
                        point_overall[metric][model_idx]
                        - point_model[metric][model_idx]
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "permutation_p": p_values[metric],
                }
            )
    add_holm(overall_comparison_rows)

    method = {
        "bootstrap_samples": args.bootstrap,
        "permutation_samples": args.permutations,
        "confidence_level": 0.95,
        "sampling_unit": "essay",
        "bootstrap_stratification": "human Overall score",
        "score_aggregation": "median of five repeated runs",
        "model_comparison": (
            "macro average across six dimensions; predictions swapped between "
            "models as an essay-level six-dimension block"
        ),
        "dimension_comparison": (
            "macro average across five models; human and model scores swapped "
            "between dimensions as an essay-level block"
        ),
        "multiple_testing": (
            "Holm correction within each comparison family and metric"
        ),
    }
    result = {
        "method": method,
        "model_macro_estimates": model_rows,
        "dimension_macro_estimates": dimension_rows,
        "model_comparisons": model_comparison_rows,
        "dimension_comparisons": dimension_comparison_rows,
        "overall_vs_dimensions": overall_comparison_rows,
    }

    result_path = args.output_dir / "analytic_dimension_statistical_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "analytic_model_macro_estimates.csv", model_rows)
    write_csv(args.output_dir / "analytic_dimension_macro_estimates.csv", dimension_rows)
    write_csv(args.output_dir / "analytic_model_comparisons.csv", model_comparison_rows)
    write_csv(
        args.output_dir / "analytic_dimension_comparisons.csv",
        dimension_comparison_rows,
    )
    write_csv(
        args.output_dir / "analytic_overall_vs_dimensions.csv",
        overall_comparison_rows,
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
