#!/usr/bin/env python3
"""Extract the selected TAACO cohesion features and add them to the JSONL data."""

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from paths import ELLIPSE_RESULTS, INTERIM_DATA_DIR, REPO_ROOT


TAACO_PATH = Path(os.getenv("TAACO_PATH", REPO_ROOT / "external" / "TAACO"))
WORK_DIR = INTERIM_DATA_DIR / "taaco_work"
TEXT_DIR = WORK_DIR / "essays"
TAACO_OUTPUT = WORK_DIR / "taaco_output.csv"

TAACO_CONFIG = {
    "sourceKeyOverlap": False,
    "sourceLSA": False,
    "sourceLDA": False,
    "sourceWord2vec": False,
    "wordsAll": True,
    "wordsContent": True,
    "wordsFunction": False,
    "wordsNoun": True,
    "wordsPronoun": True,
    "wordsArgument": True,
    "wordsVerb": False,
    "wordsAdjective": False,
    "wordsAdverb": False,
    "overlapSentence": True,
    "overlapParagraph": True,
    "overlapAdjacent": True,
    "overlapAdjacent2": False,
    "otherTTR": False,
    "otherConnectives": True,
    "otherGivenness": True,
    "overlapLSA": True,
    "overlapLDA": True,
    "overlapWord2vec": True,
    "overlapSynonym": True,
    "overlapNgrams": False,
    "outputTagged": False,
    "outputDiagnostic": False,
}

COHESION_COLUMNS = {
    "Lexical Overlap": {
        "Adjacent Content Overlap (Sentence)": "adjacent_overlap_cw_sent",
        "Adjacent Noun Overlap (Sentence)": "adjacent_overlap_noun_sent",
        "Adjacent Argument Overlap (Sentence)": "adjacent_overlap_argument_sent",
        "Adjacent Content Overlap (Paragraph)": "adjacent_overlap_cw_para",
        "Adjacent Noun Overlap (Paragraph)": "adjacent_overlap_noun_para",
        "Adjacent Argument Overlap (Paragraph)": "adjacent_overlap_argument_para",
    },
    "Semantic Overlap": {
        "LSA Similarity (Sentence)": "lsa_1_all_sent",
        "Word2Vec Similarity (Sentence)": "word2vec_1_all_sent",
        "LDA Similarity (Sentence)": "lda_1_all_sent",
        "LDA Similarity (Paragraph)": "lda_1_all_para",
        "Synonym Overlap (Noun)": "syn_overlap_sent_noun",
    },
    "Discourse Connectives": {
        "Reason and Purpose": "reason_and_purpose",
        "Opposition": "opposition",
        "Sentence Linking": "sentence_linking",
        "Addition": "addition",
        "Logical Connectives": "all_logical",
    },
    "Referential Cohesion": {
        "Content + Pronoun Overlap": "repeated_content_and_pronoun_lemmas",
    },
}


def rounded(value):
    return None if pd.isna(value) else round(float(value), 3)


def read_records() -> list[dict]:
    return [
        json.loads(line)
        for line in ELLIPSE_RESULTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare_text_files(records: list[dict]) -> None:
    # TAACO reads one text file per essay
    shutil.rmtree(TEXT_DIR, ignore_errors=True)
    TEXT_DIR.mkdir(parents=True)
    for index, record in enumerate(records):
        path = TEXT_DIR / f"essay_{index:05d}.txt"
        path.write_text(record["full_text"], encoding="utf-8")


def run_taaco() -> None:
    sys.path.insert(0, str(TAACO_PATH))
    from TAACOnoGUI import runTAACO

    # TAACO expects its own folder as the working directory
    previous_directory = Path.cwd()
    try:
        os.chdir(TAACO_PATH)
        runTAACO(str(TEXT_DIR), str(TAACO_OUTPUT), TAACO_CONFIG)
    finally:
        os.chdir(previous_directory)


def read_taaco_output() -> dict[str, dict]:
    frame = pd.read_csv(TAACO_OUTPUT)
    filename_column = next(
        (column for column in frame.columns if column.lower() == "filename"), None
    )
    if filename_column is None:
        raise ValueError("TAACO output does not contain a filename column.")

    # filenames link the TAACO rows back to the source essays
    rows = {}
    for _, row in frame.iterrows():
        rows[Path(str(row[filename_column])).name] = row.to_dict()
    return rows


def collect_features(row: dict) -> dict:
    return {
        group: {name: rounded(row.get(column)) for name, column in columns.items()}
        for group, columns in COHESION_COLUMNS.items()
    }


def main() -> None:
    records = read_records()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    prepare_text_files(records)
    run_taaco()
    taaco_rows = read_taaco_output()

    for index, record in enumerate(records):
        filename = f"essay_{index:05d}.txt"
        record.setdefault("metrics", {})["Cohesion"] = collect_features(
            taaco_rows[filename]
        )

    # replace the dataset after every row has been updated
    tmp_path = ELLIPSE_RESULTS.with_suffix(".tmp.jsonl")
    with tmp_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(ELLIPSE_RESULTS)
    print(f"Updated cohesion features for {len(records)} essays.")


if __name__ == "__main__":
    main()
