#!/usr/bin/env python3
"""Add syntactic features to the processed ELLIPSE data."""

import json

import syntactic_features as syntax
from paths import ELLIPSE_RESULTS


def organise_metrics(values: dict) -> dict:
    # convert short code names to the labels used in the dataset
    return {
        "Mean Length of Sentence (MLS)": values["MLS"],
        "Mean Length of Clause (MLC)": values["MLC"],
        "Coordinate Phrases per Sentence (CP/S)": values["CP/S"],
        "Coordinate Phrases per Clause (CP/C)": values["CP/C"],
        "Subordinate Clauses per Sentence (SC/S)": values["SC/S"],
        "Dependent Clauses per Clause (DC/C)": values["DC/C"],
        "Mean Dependency Distance (MDD)": values["MDD"],
        "Mean Length of Noun Phrase (MLNP)": values["MLNP"],
        "Nominal Modifiers per Noun Phrase": values["Nominal_Modifiers_per_NP"],
        "Prepositional Phrases per Clause (PP/C, proxy)": values["PP_per_Clause"],
        "Adverbial Modifiers per Clause": values["Adverbial_Modifiers_per_Clause"],
        "Passive Nominal Subjects per Clause": values[
            "Passive_Nominal_Subjects_per_Clause"
        ],
        "Modal Auxiliaries per Clause": values["Modal_Auxiliaries_per_Clause"],
    }


def main() -> None:
    # keep the JSON structure unchanged apart from this feature group
    records = [
        json.loads(line)
        for line in ELLIPSE_RESULTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for record in records:
        values = syntax.cal_syntax_metrics(record["full_text"])
        record.setdefault("metrics", {})["Syntactic Complexity"] = organise_metrics(
            values
        )

    # replace the dataset only after the new file is complete
    tmp_path = ELLIPSE_RESULTS.with_suffix(".tmp.jsonl")
    with tmp_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(ELLIPSE_RESULTS)
    print(f"Updated syntactic features for {len(records)} essays.")


if __name__ == "__main__":
    main()
