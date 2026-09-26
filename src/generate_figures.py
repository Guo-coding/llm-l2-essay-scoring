#!/usr/bin/env python3
"""Generate publication figures for Chapter 4 of the thesis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import ELLIPSE_RESULTS, RESULTS_DIR


OUT = RESULTS_DIR / "figures"
DATA_OUT = OUT / "data"

SUMMARY_FILE = RESULTS_DIR / "summary" / "ellipse_eval_summary.xlsx"
PREDICTION_FILE = ELLIPSE_RESULTS
FUSION_FILE = RESULTS_DIR / "fusion" / "csv" / "main_fusion.csv"

MODEL_KEYS = {
    "GPT-5.5": "openai::gpt-5.5",
    "Gemini 3": "gemini::gemini-3-flash-preview",
    "DeepSeek-V3.2": "deepseek::deepseek-chat",
    "GPT-5.4": "openai::gpt-5.4",
    "DeepSeek-V4": "deepseek::deepseek-v4-flash",
    "Gemini 3.1 Lite": "gemini::gemini-3.1-flash-lite-preview",
    "GPT-5.4 Nano": "openai::gpt-5.4-nano",
}

MAIN_MODELS = [
    "GPT-5.5",
    "Gemini 3",
    "DeepSeek-V3.2",
    "GPT-5.4",
    "DeepSeek-V4",
]

ALL_MODELS = MAIN_MODELS + ["Gemini 3.1 Lite", "GPT-5.4 Nano"]

COLORS = {
    "GPT-5.5": "#2F4B7C",
    "Gemini 3": "#A23B3B",
    "DeepSeek-V3.2": "#557A46",
    "GPT-5.4": "#B07D2B",
    "DeepSeek-V4": "#666666",
    "Gemini 3.1 Lite": "#7A5195",
    "GPT-5.4 Nano": "#8C6D5A",
}

MARKERS = {
    "GPT-5.5": "o",
    "Gemini 3": "s",
    "DeepSeek-V3.2": "^",
    "GPT-5.4": "D",
    "DeepSeek-V4": "v",
    "Gemini 3.1 Lite": "P",
    "GPT-5.4 Nano": "X",
}

LINESTYLES = {
    "GPT-5.5": "-",
    "Gemini 3": "--",
    "DeepSeek-V3.2": "-.",
    "GPT-5.4": (0, (5, 2)),
    "DeepSeek-V4": (0, (1, 1)),
    "Gemini 3.1 Lite": (0, (4, 1, 1, 1)),
    "GPT-5.4 Nano": (0, (2, 2)),
}


def set_style() -> None:
    # shared formatting for all thesis figures
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes) -> None:
    # keep the plots plain and readable in print
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    # PDF for the thesis and PNG for slides
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def annotate_value(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    offset: tuple[float, float] = (0, 6),
    size: float = 7.4,
) -> None:
    label = ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=size,
        color="#222222",
        zorder=6,
    )
    # white outline helps when lines pass behind a label
    label.set_path_effects(
        [path_effects.withStroke(linewidth=2.4, foreground="white")]
    )


def read_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_excel(SUMMARY_FILE, sheet_name="Summary")
    dimensions = pd.read_excel(SUMMARY_FILE, sheet_name="DimensionSummary")
    return summary, dimensions


def rubric_condition_figure(summary: pd.DataFrame) -> None:
    condition_names = {
        "essay_no_rubric": "No rubric",
        "essay_only": "Holistic rubric",
        "essay_plus_analytic_rubric": "Analytic condition",
    }
    records = []
    for model in MAIN_MODELS:
        model_key = MODEL_KEYS[model]
        for condition_key, condition_label in condition_names.items():
            row = summary[
                (summary["model"] == model_key)
                & (summary["condition"] == condition_key)
            ]
            if len(row) != 1:
                raise ValueError(f"Missing rubric result for {model}, {condition_key}")
            records.append(
                {
                    "Model": model,
                    "Condition": condition_label,
                    "QWK": float(row.iloc[0]["QWK_median"]),
                }
            )

    data = pd.DataFrame(records)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_OUT / "rubric_condition_qwk.csv", index=False)

    conditions = list(condition_names.values())
    x = np.arange(len(conditions))
    label_offsets = {
        "GPT-5.5": [(0, 7), (0, 7), (0, 7)],
        "Gemini 3": [(0, -11), (0, 7), (0, 7)],
        "DeepSeek-V3.2": [(0, -11), (0, 7), (0, 8)],
        "GPT-5.4": [(0, 8), (0, -11), (0, -11)],
        "DeepSeek-V4": [(0, 7), (0, 7), (0, -11)],
    }

    fig, ax = plt.subplots(figsize=(7.7, 4.7))
    for model in MAIN_MODELS:
        values = (
            data[data["Model"] == model]
            .set_index("Condition")
            .loc[conditions, "QWK"]
            .to_numpy()
        )
        ax.plot(
            x,
            values,
            label=model,
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.55,
            markersize=5.6,
            markerfacecolor="white",
            markeredgewidth=1.25,
            zorder=3,
        )
        for i, value in enumerate(values):
            annotate_value(ax, x[i], value, f"{value:.3f}", label_offsets[model][i])

    ax.set_xticks(x, conditions)
    ax.set_ylabel("Quadratic weighted kappa (QWK)")
    ax.set_ylim(0.25, 0.80)
    ax.set_yticks(np.arange(0.30, 0.81, 0.10))
    ax.margins(x=0.08)
    clean_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.3,
    )
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.12, right=0.98)
    save_figure(fig, "fig_rubric_condition_qwk")


def fusion_figure() -> None:
    source = pd.read_csv(FUSION_FILE)
    setting_names = {
        "Raw LLM median": "Raw LLM",
        "Calibrated LLM only": "Calibrated LLM",
        "LLM + all linguistic features": "LLM + linguistic\nfeatures",
    }
    rows = source[
        source["model"].isin(MAIN_MODELS)
        & source["setting"].isin(setting_names)
    ].copy()
    rows["Approach"] = rows["setting"].map(setting_names)
    data = rows[["model", "Approach", "overall_QWK"]].rename(
        columns={"model": "Model", "overall_QWK": "QWK"}
    )
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_OUT / "fusion_qwk.csv", index=False)

    approaches = list(setting_names.values())
    x = np.arange(len(approaches))
    label_offsets = {
        "GPT-5.5": [(0, 7), (-14, -11)],
        "Gemini 3": [(0, 7), (-7, 9)],
        "DeepSeek-V3.2": [(0, 7), (0, -11)],
        "GPT-5.4": [(0, -11), (13, 9)],
        "DeepSeek-V4": [(0, 7), (12, -11)],
    }
    endpoint_label_y = {
        "GPT-5.4": 0.837,
        "GPT-5.5": 0.818,
        "Gemini 3": 0.787,
        "DeepSeek-V3.2": 0.750,
        "DeepSeek-V4": 0.728,
    }

    fig, ax = plt.subplots(figsize=(7.7, 4.8))
    for model in MAIN_MODELS:
        values = (
            data[data["Model"] == model]
            .set_index("Approach")
            .loc[approaches, "QWK"]
            .to_numpy()
        )
        ax.plot(
            x,
            values,
            label=model,
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.55,
            markersize=5.6,
            markerfacecolor="white",
            markeredgewidth=1.25,
            zorder=3,
        )
        for i, value in enumerate(values[:2]):
            annotate_value(ax, x[i], value, f"{value:.3f}", label_offsets[model][i])
        ax.annotate(
            f"{values[2]:.3f}",
            (x[2], values[2]),
            xytext=(2.21, endpoint_label_y[model]),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=7.6,
            color="#222222",
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS[model],
                "linewidth": 0.7,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    reference = 0.627
    ax.axhline(reference, color="#555555", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(
        2.30,
        reference,
        "Features only  0.627",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.set_xticks(x, approaches)
    ax.set_ylabel("Quadratic weighted kappa (QWK)")
    ax.set_ylim(0.35, 0.85)
    ax.set_yticks(np.arange(0.40, 0.86, 0.10))
    ax.set_xlim(-0.12, 2.38)
    clean_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.3,
    )
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.12, right=0.98)
    save_figure(fig, "fig_fusion_qwk")


def load_baseline_predictions() -> pd.DataFrame:
    records = []
    with PREDICTION_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            essay = json.loads(line)
            human = float(essay["Overall"])
            results = essay.get("ellipse_result", {})
            for model in MAIN_MODELS:
                values = results.get(MODEL_KEYS[model], {}).get("essay_only")
                if not isinstance(values, list) or len(values) != 5:
                    raise ValueError(f"Expected five baseline scores for {model}")
                scores = np.asarray([float(value) for value in values], dtype=float)
                records.append(
                    {
                        "Essay": essay["text_id_kaggle"],
                        "Model": model,
                        "Human score": human,
                        "Median LLM score": float(np.median(scores)),
                    }
                )
    data = pd.DataFrame(records)
    data["Signed error"] = data["Median LLM score"] - data["Human score"]
    return data


def score_bias_figure(predictions: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for model in MAIN_MODELS:
        model_data = predictions[predictions["Model"] == model]
        for human_score, group in model_data.groupby("Human score", sort=True):
            errors = group["Signed error"].to_numpy(dtype=float)
            bootstrap_means = np.empty(5000, dtype=float)
            # bootstrap MSD within each human score level
            for iteration in range(5000):
                bootstrap_means[iteration] = rng.choice(
                    errors, size=len(errors), replace=True
                ).mean()
            rows.append(
                {
                    "Model": model,
                    "Human score": human_score,
                    "Essays": len(errors),
                    "Mean signed error": errors.mean(),
                    "CI lower": np.quantile(bootstrap_means, 0.025),
                    "CI upper": np.quantile(bootstrap_means, 0.975),
                }
            )
    data = pd.DataFrame(rows)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_OUT / "bias_by_human_score.csv", index=False)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.0, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    score_levels = np.arange(1.0, 5.01, 0.5)
    for panel_index, model in enumerate(MAIN_MODELS):
        ax = axes.flat[panel_index]
        frame = data[data["Model"] == model].sort_values("Human score")
        x = frame["Human score"].to_numpy()
        y = frame["Mean signed error"].to_numpy()
        lower = y - frame["CI lower"].to_numpy()
        upper = frame["CI upper"].to_numpy() - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle="-",
            linewidth=1.35,
            elinewidth=0.8,
            capsize=2.0,
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
        for point_index, (xi, yi) in enumerate(zip(x, y)):
            vertical_offset = (0, 7) if point_index % 2 == 0 else (0, -10)
            annotate_value(
                ax,
                xi,
                yi,
                f"{yi:+.2f}",
                vertical_offset,
                size=6.6,
            )
        ax.axhline(
            0, color="#333333", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1
        )
        ax.set_title(model, color=COLORS[model], fontweight="semibold", pad=5)
        ax.set_xticks(score_levels, [f"{score:.1f}" for score in score_levels])
        ax.set_xlim(0.86, 5.14)
        ax.set_ylim(-2.16, 0.80)
        ax.set_yticks(np.arange(-2.0, 0.81, 0.5))
        clean_axes(ax)

    axes.flat[-1].axis("off")
    fig.supxlabel("Human score", fontsize=10)
    fig.supylabel("Mean signed error (LLM score − human score)", fontsize=10)
    save_figure(fig, "fig_bias_by_human_score")


def stability_agreement_figure(summary: pd.DataFrame) -> None:
    rows = []
    for model in ALL_MODELS:
        row = summary[
            (summary["model"] == MODEL_KEYS[model])
            & (summary["condition"] == "essay_only")
        ]
        if len(row) != 1:
            raise ValueError(f"Missing stability result for {model}")
        rows.append(
            {
                "Model": model,
                "ICC": float(row.iloc[0]["ICC_runs"]),
                "QWK": float(row.iloc[0]["QWK_median"]),
            }
        )
    data = pd.DataFrame(rows)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_OUT / "stability_and_agreement.csv", index=False)

    text_positions = {
        "GPT-5.5": (0.865, 0.686),
        "Gemini 3": (0.835, 0.620),
        "DeepSeek-V3.2": (0.790, 0.565),
        "GPT-5.4": (0.835, 0.485),
        "DeepSeek-V4": (0.710, 0.430),
        "Gemini 3.1 Lite": (0.830, 0.530),
        "GPT-5.4 Nano": (0.660, 0.282),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for _, row in data.iterrows():
        model = row["Model"]
        ax.scatter(
            row["ICC"],
            row["QWK"],
            s=48,
            marker=MARKERS[model],
            facecolor="white",
            edgecolor=COLORS[model],
            linewidth=1.45,
            zorder=3,
        )
        text_x, text_y = text_positions[model]
        label = ax.annotate(
            f"{model}\n({row['ICC']:.3f}, {row['QWK']:.3f})",
            (row["ICC"], row["QWK"]),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=7.5,
            ha="left",
            va="center",
            color="#222222",
            arrowprops={
                "arrowstyle": "-",
                "color": "#888888",
                "linewidth": 0.65,
                "shrinkA": 3,
                "shrinkB": 4,
            },
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.5, foreground="white")]
        )

    ax.set_xlabel("Intraclass correlation coefficient (ICC)")
    ax.set_ylabel("Quadratic weighted kappa (QWK)")
    ax.set_xlim(0.60, 0.985)
    ax.set_ylim(0.26, 0.70)
    ax.set_xticks(np.arange(0.60, 1.00, 0.05))
    ax.set_yticks(np.arange(0.30, 0.71, 0.10))
    clean_axes(ax)
    fig.subplots_adjust(top=0.97, bottom=0.14, left=0.13, right=0.98)
    save_figure(fig, "fig_stability_vs_agreement")


def analytic_heatmap_figure(dimensions: pd.DataFrame) -> None:
    dimension_order = [
        "Cohesion",
        "Syntax",
        "Vocabulary",
        "Phraseology",
        "Grammar",
        "Conventions",
    ]
    rows = dimensions[
        (dimensions["condition"] == "essay_plus_analytic_rubric")
        & dimensions["model"].isin([MODEL_KEYS[name] for name in MAIN_MODELS])
        & dimensions["dimension"].isin(dimension_order)
    ].copy()
    reverse_names = {value: key for key, value in MODEL_KEYS.items()}
    rows["Model"] = rows["model"].map(reverse_names)
    matrix = (
        rows.pivot(index="Model", columns="dimension", values="QWK_median")
        .loc[MAIN_MODELS, dimension_order]
        .astype(float)
    )
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(DATA_OUT / "analytic_dimension_qwk.csv")

    heatmap_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "thesis_teal",
        ["#F4F8F7", "#D5E9E5", "#8FC2BD", "#4A8995", "#174B63"],
    )

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    image = ax.pcolormesh(
        np.arange(matrix.shape[1] + 1),
        np.arange(matrix.shape[0] + 1),
        matrix.to_numpy(),
        cmap=heatmap_cmap,
        vmin=0.20,
        vmax=0.70,
        shading="flat",
        edgecolors="white",
        linewidth=0.9,
    )
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_xticks(np.arange(len(dimension_order)) + 0.5, dimension_order)
    ax.set_yticks(np.arange(len(MAIN_MODELS)) + 0.5, MAIN_MODELS)
    ax.tick_params(axis="x", rotation=0, length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=7)
    for label in ax.get_xticklabels():
        label.set_ha("center")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            colour = "white" if value >= 0.54 else "#18323A"
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9.4,
                fontweight="semibold",
                color=colour,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    colour_bar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
    colour_bar.set_label("QWK", rotation=90)
    colour_bar.outline.set_linewidth(0.7)
    colour_bar.set_ticks(np.arange(0.20, 0.71, 0.10))
    fig.subplots_adjust(top=0.98, bottom=0.16, left=0.18, right=0.94)
    save_figure(fig, "fig_analytic_dimension_heatmap")


def main() -> None:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    summary, dimensions = read_summary()
    rubric_condition_figure(summary)
    fusion_figure()
    predictions = load_baseline_predictions()
    score_bias_figure(predictions)
    stability_agreement_figure(summary)
    analytic_heatmap_figure(dimensions)

    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
