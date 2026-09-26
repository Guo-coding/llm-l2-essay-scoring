#!/usr/bin/env python3
"""Add readability and lexical features to the processed ELLIPSE data."""

import json

import lexical_features as lexical
from paths import ELLIPSE_RESULTS


def organise_metrics(values: dict) -> tuple[dict, dict]:
    # labels used in the thesis tables
    readability = {
        "Flesch Reading Ease": values["flesch_reading_ease"],
        "Flesch–Kincaid Grade Level": values["flesch_kincaid_grade"],
        "Gunning Fog Index": values["gunning_fog"],
        "Automated Readability Index": values["ari"],
    }
    lexical_complexity = {
        "Total Tokens": values["total_tokens"],
        "Lexical Diversity": {
            "MTLD": values["mtld"],
            "MATTR": values["mattr"],
            "HD-D": values["hdd"],
        },
        "Lexical Density (Content Words / All Tokens)": values["lexical_density"],
        "Lexical Sophistication": {
            "High-Frequency Word Coverage (Top 3000)": values["high_freq_coverage"],
            "Low-Frequency Word Ratio (Beyond Top 3000, In-list Only)": values[
                "low_frequency_ratio"
            ],
            "Mean Log Word Frequency (In-list Only)": values["mean_log_frequency"],
        },
        "Psycholinguistic Norms": {
            "Mean Word Concreteness": values["avg_word_concreteness"],
            "Mean Age of Acquisition (AoA)": values["mean_aoa"],
        },
    }
    return readability, lexical_complexity


def main() -> None:
    # read once, add both feature groups, then write once
    records = [
        json.loads(line)
        for line in ELLIPSE_RESULTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for record in records:
        values = lexical.compute_lexical_metrics(record["full_text"])
        readability, lexical_complexity = organise_metrics(values)
        metrics = record.setdefault("metrics", {})
        metrics["Readability"] = readability
        metrics["Lexical Complexity"] = lexical_complexity

    # finish the new file before replacing the old one
    tmp_path = ELLIPSE_RESULTS.with_suffix(".tmp.jsonl")
    with tmp_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(ELLIPSE_RESULTS)
    print(f"Updated readability and lexical features for {len(records)} essays.")


if __name__ == "__main__":
    main()
