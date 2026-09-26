#!/usr/bin/env python3
"""Run Ridge calibration, feature fusion, and feature-group ablation."""

from __future__ import annotations

import json
import math
import statistics
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from paths import ELLIPSE_RESULTS, RESULTS_DIR, display_path


INPUT_PATH = ELLIPSE_RESULTS
OUTPUT_DIR = RESULTS_DIR / "fusion" / "csv"
OUTPUT_XLSX = RESULTS_DIR / "fusion" / "fusion_ablation_results.xlsx"

TARGET_SCORE = "Overall"
LLM_CONDITION = "essay_only"
RANDOM_STATE = 42
N_SPLITS = 5
VALID_MIN = 1.0
VALID_MAX = 5.0
SCORE_STEP = 0.5

MODELS = [
    ("GPT-5.5", "openai::gpt-5.5"),
    ("Gemini 3", "gemini::gemini-3-flash-preview"),
    ("DeepSeek-V3.2", "deepseek::deepseek-chat"),
    ("GPT-5.4", "openai::gpt-5.4"),
    ("DeepSeek-V4", "deepseek::deepseek-v4-flash"),
]

FEATURE_GROUPS = [
    ("Lexical", "Lexical Complexity"),
    ("Readability", "Readability"),
    ("Syntactic", "Syntactic Complexity"),
    ("Cohesion", "Cohesion"),
]

ALPHAS = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
SCORE_LABELS = np.arange(int(VALID_MIN / SCORE_STEP), int(VALID_MAX / SCORE_STEP) + 1)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._base")


def flatten_metrics(d: dict, prefix: str = "") -> dict[str, float]:
    # turn the nested metric groups into model columns
    flat: dict[str, float] = {}
    for key, value in d.items():
        new_key = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, new_key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[new_key] = float(value)
    return flat


def clean_score(value) -> float | None:
    # API errors and out-of-range values are ignored here
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or x < VALID_MIN or x > VALID_MAX:
        return None
    return x


def round_to_half(values) -> np.ndarray:
    # return predictions to the ELLIPSE half-point scale
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, VALID_MIN, VALID_MAX)
    return np.round(arr / SCORE_STEP) * SCORE_STEP


def adjacent_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) <= SCORE_STEP))


def evaluate(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_cls = np.rint(y_true / SCORE_STEP).astype(int)
    y_pred_cls = np.rint(y_pred / SCORE_STEP).astype(int)
    rho = spearmanr(y_true, y_pred).correlation
    if rho is None or math.isnan(float(rho)):
        rho = np.nan
    return {
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred),
        "QWK": float(cohen_kappa_score(y_true_cls, y_pred_cls, labels=SCORE_LABELS, weights="quadratic")),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSD": float(np.mean(y_pred - y_true)),
        "Spearman": float(rho),
    }


def fmt_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return round(float(value), 3)


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    feature_rows = []
    seen_features = set()

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gold = clean_score(obj.get(TARGET_SCORE))
            if gold is None:
                continue

            # feature names are collected once for the ablation groups
            metrics = flatten_metrics(obj.get("metrics", {}))
            for feature_name in metrics:
                if feature_name not in seen_features:
                    seen_features.add(feature_name)
                    feature_rows.append(
                        {
                            "feature_group": feature_name.split("__", 1)[0],
                            "feature_name": feature_name,
                        }
                    )

            base = {
                "essay_id": obj.get("text_id_kaggle"),
                "gold": gold,
            }

            model_block = obj.get("ellipse_result", {})
            for model_name, model_key in MODELS:
                scores_raw = model_block.get(model_key, {}).get(LLM_CONDITION, [])
                scores = [clean_score(x) for x in scores_raw]
                scores = [x for x in scores if x is not None]
                if not scores:
                    continue

                row = dict(base)
                row.update(
                    {
                        "model": model_name,
                        "model_key": model_key,
                        "llm_median": statistics.median(scores),
                        "llm_mean": statistics.mean(scores),
                        "llm_sd": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                        "n_runs": len(scores),
                    }
                )
                row.update(metrics)
                records.append(row)

    return pd.DataFrame(records), pd.DataFrame(feature_rows).sort_values(["feature_group", "feature_name"])


def ridge_model() -> Pipeline:
    # preprocessing stays inside each training fold
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS, cv=5)),
        ]
    )


def summarize_fold_metrics(rows: list[dict], overall_metrics: dict, alpha_values: list[float]) -> dict:
    by_metric = defaultdict(list)
    for row in rows:
        for key in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            by_metric[key].append(row[key])

    summary = {
        "n_essays": sum(row["n_essays"] for row in rows),
        "cv": f"{N_SPLITS}-fold stratified",
        "mean_alpha": fmt_float(np.mean(alpha_values)) if alpha_values else "",
        "features": "",
        "note": "",
    }
    for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
        vals = np.asarray(by_metric[metric], dtype=float)
        summary[f"overall_{metric}"] = fmt_float(overall_metrics[metric])
        summary[f"mean_{metric}"] = fmt_float(np.nanmean(vals))
        summary[f"sd_{metric}"] = fmt_float(np.nanstd(vals, ddof=1))
    return summary


def run_ridge_cv(X: pd.DataFrame, y: np.ndarray, strata: np.ndarray) -> tuple[np.ndarray, list[dict], list[float]]:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(X), np.nan)
    fold_metrics = []
    alpha_values = []

    # one held-out prediction for each essay
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, strata), start=1):
        model = ridge_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        pred_cont = model.predict(X.iloc[test_idx])
        pred = round_to_half(pred_cont)

        oof[test_idx] = pred
        alpha_values.append(float(model.named_steps["ridge"].alpha_))

        metrics = evaluate(y[test_idx], pred)
        fold_metrics.append({"fold": fold, "n_essays": len(test_idx), **metrics})

    return oof, fold_metrics, alpha_values


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_df = load_rows()
    if df.empty:
        raise RuntimeError("No usable LLM score rows were loaded.")

    metric_cols = feature_df["feature_name"].tolist()
    feature_group_cols = {
        group_label: feature_df.loc[feature_df["feature_group"] == group_name, "feature_name"].tolist()
        for group_label, group_name in FEATURE_GROUPS
    }
    missing_groups = [group_label for group_label, cols in feature_group_cols.items() if not cols]
    if missing_groups:
        raise RuntimeError(f"Missing feature columns for: {', '.join(missing_groups)}")

    main_rows = []
    fold_rows = []
    prediction_rows = []
    change_rows = []
    subset_rows = []
    add_one_rows = []
    leave_one_rows = []

    for model_name, model_key in MODELS:
        model_df = df[df["model_key"] == model_key].copy().reset_index(drop=True)
        if len(model_df) != 1200:
            print(f"Warning: {model_name} has {len(model_df)} usable rows.")

        y = model_df["gold"].astype(float).to_numpy()
        strata = model_df["gold"].astype(str).to_numpy()
        X_llm = model_df[["llm_median"]].astype(float)
        X_linguistic = model_df[metric_cols].astype(float)
        X_fusion = model_df[["llm_median"] + metric_cols].astype(float)

        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        raw_fold_metrics = []
        cal_fold_metrics = []
        ling_fold_metrics = []
        cal_oof = np.full(len(model_df), np.nan)
        cal_cont_oof = np.full(len(model_df), np.nan)
        ling_oof = np.full(len(model_df), np.nan)
        ling_cont_oof = np.full(len(model_df), np.nan)
        fusion_oof = np.full(len(model_df), np.nan)
        fusion_cont_oof = np.full(len(model_df), np.nan)
        raw_median_oof = np.full(len(model_df), np.nan)
        alpha_values = []
        ling_alpha_values = []
        fusion_alpha_values = []

        # use the same folds for the raw-score summaries
        for fold, (_, test_idx) in enumerate(splitter.split(X_llm, strata), start=1):
            median_pred = round_to_half(model_df.loc[test_idx, "llm_median"].to_numpy())
            raw_median_oof[test_idx] = median_pred
            metrics = evaluate(y[test_idx], median_pred)
            raw_fold_metrics.append({"fold": fold, "n_essays": len(test_idx), **metrics})
            fold_rows.append(
                {
                    "model": model_name,
                    "model_key": model_key,
                    "setting": "Raw LLM median",
                    "fold": fold,
                    "n_essays": len(test_idx),
                    "ridge_alpha": "",
                    **{k: fmt_float(v) for k, v in metrics.items()},
                }
            )

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_llm, strata), start=1):
            model = ridge_model()
            model.fit(X_llm.iloc[train_idx], y[train_idx])
            pred_cont = model.predict(X_llm.iloc[test_idx])
            pred = round_to_half(pred_cont)

            cal_cont_oof[test_idx] = pred_cont
            cal_oof[test_idx] = pred
            alpha = float(model.named_steps["ridge"].alpha_)
            alpha_values.append(alpha)

            metrics = evaluate(y[test_idx], pred)
            cal_fold_metrics.append({"fold": fold, "n_essays": len(test_idx), **metrics})
            fold_rows.append(
                {
                    "model": model_name,
                    "model_key": model_key,
                    "setting": "Calibrated LLM only",
                    "fold": fold,
                    "n_essays": len(test_idx),
                    "ridge_alpha": fmt_float(alpha),
                    **{k: fmt_float(v) for k, v in metrics.items()},
                }
            )

            ling_model = ridge_model()
            ling_model.fit(X_linguistic.iloc[train_idx], y[train_idx])
            ling_pred_cont = ling_model.predict(X_linguistic.iloc[test_idx])
            ling_pred = round_to_half(ling_pred_cont)

            ling_cont_oof[test_idx] = ling_pred_cont
            ling_oof[test_idx] = ling_pred
            ling_alpha = float(ling_model.named_steps["ridge"].alpha_)
            ling_alpha_values.append(ling_alpha)

            metrics = evaluate(y[test_idx], ling_pred)
            ling_fold_metrics.append({"fold": fold, "n_essays": len(test_idx), **metrics})
            fold_rows.append(
                {
                    "model": model_name,
                    "model_key": model_key,
                    "setting": "Linguistic features only",
                    "fold": fold,
                    "n_essays": len(test_idx),
                    "ridge_alpha": fmt_float(ling_alpha),
                    **{k: fmt_float(v) for k, v in metrics.items()},
                }
            )

            fusion_model = ridge_model()
            fusion_model.fit(X_fusion.iloc[train_idx], y[train_idx])
            fusion_pred_cont = fusion_model.predict(X_fusion.iloc[test_idx])
            fusion_pred = round_to_half(fusion_pred_cont)

            fusion_cont_oof[test_idx] = fusion_pred_cont
            fusion_oof[test_idx] = fusion_pred
            fusion_alpha = float(fusion_model.named_steps["ridge"].alpha_)
            fusion_alpha_values.append(fusion_alpha)

            metrics = evaluate(y[test_idx], fusion_pred)
            fold_rows.append(
                {
                    "model": model_name,
                    "model_key": model_key,
                    "setting": "LLM + all linguistic features",
                    "fold": fold,
                    "n_essays": len(test_idx),
                    "ridge_alpha": fmt_float(fusion_alpha),
                    **{k: fmt_float(v) for k, v in metrics.items()},
                }
            )

            for local_idx, original_idx in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "essay_id": model_df.loc[original_idx, "essay_id"],
                        "model": model_name,
                        "model_key": model_key,
                        "fold": fold,
                        "gold": model_df.loc[original_idx, "gold"],
                        "llm_median": fmt_float(model_df.loc[original_idx, "llm_median"]),
                        "llm_sd": fmt_float(model_df.loc[original_idx, "llm_sd"]),
                        "raw_median_pred": fmt_float(raw_median_oof[original_idx]),
                        "calibrated_continuous": fmt_float(pred_cont[local_idx]),
                        "calibrated_pred": fmt_float(pred[local_idx]),
                        "linguistic_only_continuous": fmt_float(ling_pred_cont[local_idx]),
                        "linguistic_only_pred": fmt_float(ling_pred[local_idx]),
                        "fusion_continuous": fmt_float(fusion_pred_cont[local_idx]),
                        "fusion_pred": fmt_float(fusion_pred[local_idx]),
                    }
                )

        raw_median_overall = evaluate(y, raw_median_oof)
        cal_overall = evaluate(y, cal_oof)
        ling_overall = evaluate(y, ling_oof)
        fusion_overall = evaluate(y, fusion_oof)

        subset_options = [((), "LLM only")]
        group_labels = [group_label for group_label, _ in FEATURE_GROUPS]

        # try every feature-group combination used in the ablation tables
        for subset_size in range(1, len(group_labels) + 1):
            for subset in combinations(group_labels, subset_size):
                subset_options.append((subset, " + ".join(subset)))

        for subset, subset_label in subset_options:
            subset_cols = []
            for feature_label in subset:
                subset_cols.extend(feature_group_cols[feature_label])

            X_subset = model_df[["llm_median"] + subset_cols].astype(float)
            subset_oof, subset_fold_metrics, subset_alpha_values = run_ridge_cv(X_subset, y, strata)
            subset_overall = evaluate(y, subset_oof)
            subset_summary = summarize_fold_metrics(
                subset_fold_metrics,
                subset_overall,
                subset_alpha_values,
            )

            if subset:
                setting_name = f"LLM + {subset_label}"
                features_name = "llm_median + " + subset_label
            else:
                setting_name = "LLM only"
                features_name = "llm_median"

            subset_summary.update(
                {
                    "model": model_name,
                    "model_key": model_key,
                    "setting": setting_name,
                    "estimator": "RidgeCV",
                    "features": features_name,
                    "feature_groups": subset_label,
                    "n_feature_groups": len(subset),
                    "n_linguistic_features": len(subset_cols),
                    "included_lexical": "Yes" if "Lexical" in subset else "No",
                    "included_readability": "Yes" if "Readability" in subset else "No",
                    "included_syntactic": "Yes" if "Syntactic" in subset else "No",
                    "included_cohesion": "Yes" if "Cohesion" in subset else "No",
                    "note": "Subset ablation: median LLM score plus one selected combination of linguistic feature groups.",
                }
            )
            for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
                subset_summary[f"delta_vs_calibrated_{metric}"] = fmt_float(
                    subset_overall[metric] - cal_overall[metric]
                )
                subset_summary[f"delta_vs_all_features_{metric}"] = fmt_float(
                    subset_overall[metric] - fusion_overall[metric]
                )
                subset_summary[f"delta_vs_raw_median_{metric}"] = fmt_float(
                    subset_overall[metric] - raw_median_overall[metric]
                )
            subset_rows.append(subset_summary)

        for feature_label, _ in FEATURE_GROUPS:
            group_cols = feature_group_cols[feature_label]
            X_add_one = model_df[["llm_median"] + group_cols].astype(float)
            add_one_oof = np.full(len(model_df), np.nan)
            add_one_fold_metrics = []
            add_one_alpha_values = []

            for fold, (train_idx, test_idx) in enumerate(splitter.split(X_add_one, strata), start=1):
                add_one_model = ridge_model()
                add_one_model.fit(X_add_one.iloc[train_idx], y[train_idx])
                pred_cont = add_one_model.predict(X_add_one.iloc[test_idx])
                pred = round_to_half(pred_cont)

                add_one_oof[test_idx] = pred
                alpha = float(add_one_model.named_steps["ridge"].alpha_)
                add_one_alpha_values.append(alpha)

                metrics = evaluate(y[test_idx], pred)
                add_one_fold_metrics.append({"fold": fold, "n_essays": len(test_idx), **metrics})

            add_one_overall = evaluate(y, add_one_oof)
            add_one_summary = summarize_fold_metrics(
                add_one_fold_metrics,
                add_one_overall,
                add_one_alpha_values,
            )
            add_one_summary.update(
                {
                    "model": model_name,
                    "model_key": model_key,
                    "setting": f"LLM + {feature_label.lower()} features",
                    "estimator": "RidgeCV",
                    "features": f"llm_median + {feature_label}",
                    "feature_group": feature_label,
                    "n_linguistic_features": len(group_cols),
                    "note": "Add-one ablation: median LLM score plus one linguistic feature group.",
                }
            )
            for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
                add_one_summary[f"delta_vs_calibrated_{metric}"] = fmt_float(
                    add_one_overall[metric] - cal_overall[metric]
                )
                add_one_summary[f"delta_vs_raw_median_{metric}"] = fmt_float(
                    add_one_overall[metric] - raw_median_overall[metric]
                )
            add_one_rows.append(add_one_summary)

        for feature_label, _ in FEATURE_GROUPS:
            removed_cols = set(feature_group_cols[feature_label])
            remaining_cols = [col for col in metric_cols if col not in removed_cols]
            X_leave_one = model_df[["llm_median"] + remaining_cols].astype(float)
            leave_one_oof = np.full(len(model_df), np.nan)
            leave_one_fold_metrics = []
            leave_one_alpha_values = []

            for fold, (train_idx, test_idx) in enumerate(splitter.split(X_leave_one, strata), start=1):
                leave_one_model = ridge_model()
                leave_one_model.fit(X_leave_one.iloc[train_idx], y[train_idx])
                pred_cont = leave_one_model.predict(X_leave_one.iloc[test_idx])
                pred = round_to_half(pred_cont)

                leave_one_oof[test_idx] = pred
                alpha = float(leave_one_model.named_steps["ridge"].alpha_)
                leave_one_alpha_values.append(alpha)

                metrics = evaluate(y[test_idx], pred)
                leave_one_fold_metrics.append({"fold": fold, "n_essays": len(test_idx), **metrics})

            leave_one_overall = evaluate(y, leave_one_oof)
            leave_one_summary = summarize_fold_metrics(
                leave_one_fold_metrics,
                leave_one_overall,
                leave_one_alpha_values,
            )
            leave_one_summary.update(
                {
                    "model": model_name,
                    "model_key": model_key,
                    "setting": f"All except {feature_label.lower()} features",
                    "estimator": "RidgeCV",
                    "features": f"llm_median + all features except {feature_label}",
                    "removed_feature_group": feature_label,
                    "n_removed_features": len(removed_cols),
                    "n_linguistic_features_used": len(remaining_cols),
                    "note": "Leave-one-out ablation: median LLM score plus all linguistic features except one removed group.",
                }
            )
            for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
                leave_one_summary[f"delta_vs_all_features_{metric}"] = fmt_float(
                    leave_one_overall[metric] - fusion_overall[metric]
                )
                leave_one_summary[f"delta_vs_calibrated_{metric}"] = fmt_float(
                    leave_one_overall[metric] - cal_overall[metric]
                )
                leave_one_summary[f"delta_vs_raw_median_{metric}"] = fmt_float(
                    leave_one_overall[metric] - raw_median_overall[metric]
                )
            leave_one_rows.append(leave_one_summary)

        raw_summary = summarize_fold_metrics(raw_fold_metrics, raw_median_overall, [])
        raw_summary.update(
            {
                "model": model_name,
                "model_key": model_key,
                "setting": "Raw LLM median",
                "estimator": "No calibration",
                "features": "llm_median",
                "note": "Median of five essay_only runs.",
            }
        )
        main_rows.append(raw_summary)

        cal_summary = summarize_fold_metrics(cal_fold_metrics, cal_overall, alpha_values)
        cal_summary.update(
            {
                "model": model_name,
                "model_key": model_key,
                "setting": "Calibrated LLM only",
                "estimator": "RidgeCV",
                "features": "llm_median",
                "note": "Out-of-fold RidgeCV predictions, clipped to 1.0-5.0 and rounded to nearest 0.5.",
            }
        )
        main_rows.append(cal_summary)

        ling_summary = summarize_fold_metrics(ling_fold_metrics, ling_overall, ling_alpha_values)
        ling_summary.update(
            {
                "model": model_name,
                "model_key": model_key,
                "setting": "Linguistic features only",
                "estimator": "RidgeCV",
                "features": "all_linguistic_features",
                "note": "Out-of-fold RidgeCV predictions using only linguistic features; no LLM scores included.",
            }
        )
        main_rows.append(ling_summary)

        fusion_fold_metrics = [
            row for row in fold_rows if row["model_key"] == model_key and row["setting"] == "LLM + all linguistic features"
        ]
        fusion_summary = summarize_fold_metrics(fusion_fold_metrics, fusion_overall, fusion_alpha_values)
        fusion_summary.update(
            {
                "model": model_name,
                "model_key": model_key,
                "setting": "LLM + all linguistic features",
                "estimator": "RidgeCV",
                "features": "llm_median + all_linguistic_features",
                "note": "Out-of-fold RidgeCV predictions using the median LLM score and all linguistic features.",
            }
        )
        main_rows.append(fusion_summary)

        change = {"model": model_name, "model_key": model_key, "comparison": "Calibrated LLM only - Raw LLM median"}
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            change[f"delta_{metric}"] = fmt_float(cal_overall[metric] - raw_median_overall[metric])
        change_rows.append(change)

        change = {"model": model_name, "model_key": model_key, "comparison": "Linguistic features only - Raw LLM median"}
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            change[f"delta_{metric}"] = fmt_float(ling_overall[metric] - raw_median_overall[metric])
        change_rows.append(change)

        change = {"model": model_name, "model_key": model_key, "comparison": "Linguistic features only - Calibrated LLM only"}
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            change[f"delta_{metric}"] = fmt_float(ling_overall[metric] - cal_overall[metric])
        change_rows.append(change)

        change = {"model": model_name, "model_key": model_key, "comparison": "LLM + all linguistic features - Calibrated LLM only"}
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            change[f"delta_{metric}"] = fmt_float(fusion_overall[metric] - cal_overall[metric])
        change_rows.append(change)

        change = {"model": model_name, "model_key": model_key, "comparison": "LLM + all linguistic features - Raw LLM median"}
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            change[f"delta_{metric}"] = fmt_float(fusion_overall[metric] - raw_median_overall[metric])
        change_rows.append(change)

        change = {"model": model_name, "model_key": model_key, "comparison": "LLM + all linguistic features - Linguistic features only"}
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            change[f"delta_{metric}"] = fmt_float(fusion_overall[metric] - ling_overall[metric])
        change_rows.append(change)

    main_cols = [
        "model",
        "model_key",
        "setting",
        "estimator",
        "features",
        "n_essays",
        "cv",
        "overall_accuracy",
        "mean_accuracy",
        "sd_accuracy",
        "overall_adjacent_accuracy",
        "mean_adjacent_accuracy",
        "sd_adjacent_accuracy",
        "overall_QWK",
        "mean_QWK",
        "sd_QWK",
        "overall_MAE",
        "mean_MAE",
        "sd_MAE",
        "overall_MSD",
        "mean_MSD",
        "sd_MSD",
        "overall_Spearman",
        "mean_Spearman",
        "sd_Spearman",
        "mean_alpha",
        "note",
    ]
    delta_cols = []
    for baseline_name in ["calibrated", "raw_median"]:
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            delta_cols.append(f"delta_vs_{baseline_name}_{metric}")
    add_one_cols = (
        main_cols[:-1]
        + ["feature_group", "n_linguistic_features"]
        + delta_cols
        + ["note"]
    )
    subset_delta_cols = []
    for baseline_name in ["calibrated", "all_features", "raw_median"]:
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            subset_delta_cols.append(f"delta_vs_{baseline_name}_{metric}")
    subset_cols = (
        main_cols[:-1]
        + [
            "feature_groups",
            "n_feature_groups",
            "n_linguistic_features",
            "included_lexical",
            "included_readability",
            "included_syntactic",
            "included_cohesion",
        ]
        + subset_delta_cols
        + ["note"]
    )
    leave_one_delta_cols = []
    for baseline_name in ["all_features", "calibrated", "raw_median"]:
        for metric in ["accuracy", "adjacent_accuracy", "QWK", "MAE", "MSD", "Spearman"]:
            leave_one_delta_cols.append(f"delta_vs_{baseline_name}_{metric}")
    leave_one_cols = (
        main_cols[:-1]
        + ["removed_feature_group", "n_removed_features", "n_linguistic_features_used"]
        + leave_one_delta_cols
        + ["note"]
    )
    readme_rows = [
        {"field": "experiment", "value": "Fusion and ablation experiments for LLM holistic score calibration"},
        {"field": "source_file", "value": display_path(INPUT_PATH)},
        {"field": "target_score", "value": TARGET_SCORE},
        {"field": "llm_condition", "value": LLM_CONDITION},
        {"field": "llm_score_used", "value": "Median of five repeated LLM scores"},
        {"field": "current_completed_part", "value": "Raw LLM median, Calibrated LLM only, Linguistic features only, LLM + all linguistic features, feature subset ablation, add-one ablation, and leave-one-out ablation"},
        {"field": "main_model", "value": "RidgeCV"},
        {"field": "validation", "value": f"{N_SPLITS}-fold stratified cross validation by human Overall score"},
        {"field": "rounding", "value": "Predictions clipped to 1.0-5.0 and rounded to nearest 0.5"},
        {"field": "main_comparison_for_later", "value": "LLM median + all linguistic features vs Calibrated LLM only"},
        {"field": "feature_subset_ablation", "value": "Each model uses llm_median plus every possible feature group subset, from no linguistic group to all four groups"},
        {"field": "add_one_ablation", "value": "Each model uses llm_median plus one feature group: lexical, readability, syntactic, or cohesion"},
        {"field": "leave_one_out_ablation", "value": "Each model uses llm_median plus all linguistic feature groups except one removed group"},
    ]
    sheets = {
        "README": pd.DataFrame(readme_rows),
        "main_fusion": pd.DataFrame(main_rows)[main_cols],
        "fusion_changes": pd.DataFrame(change_rows),
        "fold_results": pd.DataFrame(fold_rows),
        "feature_subset_ablation": pd.DataFrame(subset_rows)[subset_cols],
        "add_one_ablation": pd.DataFrame(add_one_rows)[add_one_cols],
        "leave_one_out_ablation": pd.DataFrame(leave_one_rows)[leave_one_cols],
        "predictions": pd.DataFrame(prediction_rows),
        "feature_groups": feature_df,
    }

    for sheet_name, sheet_df in sheets.items():
        sheet_df.to_csv(OUTPUT_DIR / f"{sheet_name}.csv", index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                header = str(col[0].value) if col[0].value is not None else ""
                width = min(max(len(header) + 2, 12), 32)
                ws.column_dimensions[col[0].column_letter].width = width

    print(f"Saved fusion and ablation results to {OUTPUT_XLSX}.")


if __name__ == "__main__":
    run()
