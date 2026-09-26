#!/usr/bin/env python3
"""Readability and lexical measures used in the ELLIPSE experiments."""

import math
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import spacy
import textstat
from lexicalrichness import LexicalRichness

from paths import RESOURCE_DIR


nlp = spacy.load("en_core_web_sm")
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def parse_text(text: str):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    return tokens, doc


@lru_cache(maxsize=1)
def load_freq_list(file_path: str = str(RESOURCE_DIR / "bnc_wordfreq.csv")):
    # the same lookup table is reused for all essays
    frame = pd.read_csv(file_path)
    words = frame["word"].astype(str).str.lower()
    frequency = dict(zip(words, frame["freq"]))
    rank = dict(zip(words, frame["rank"]))
    return frequency, rank


@lru_cache(maxsize=1)
def load_concreteness_dict(path: str):
    frame = pd.read_csv(path)
    return dict(
        zip(frame["word"].astype(str).str.lower(), frame["conc"].astype(float))
    )


@lru_cache(maxsize=1)
def load_aoa_dict(path: str, preferred_column: str = "AoA_Kup"):
    frame = pd.read_excel(path)
    # use the Kuperman ratings when the original column name is available
    if preferred_column in frame.columns:
        aoa_column = preferred_column
    else:
        aoa_column = next(column for column in frame.columns if column.startswith("AoA_"))
    values = frame[["Word", aoa_column]].dropna()
    return dict(
        zip(
            values["Word"].astype(str).str.lower(),
            values[aoa_column].astype(float),
        )
    )


def high_freq_coverage(tokens: list[str], ranks: dict, top_n: int = 3000):
    if not tokens:
        return np.nan
    matches = sum(ranks.get(token.lower(), math.inf) <= top_n for token in tokens)
    return matches / len(tokens)


def low_frequency_ratio(tokens: list[str], ranks: dict, top_n: int = 3000):
    if not tokens:
        return np.nan
    matches = sum(
        token.lower() in ranks and ranks[token.lower()] > top_n for token in tokens
    )
    return matches / len(tokens)


def average_log_frequency(tokens: list[str], frequencies: dict):
    values = [
        math.log10(frequencies[token.lower()] + 1)
        for token in tokens
        if token.lower() in frequencies
    ]
    return float(np.mean(values)) if values else np.nan


def lexical_density(doc):
    tags = [
        token.pos_ for token in doc if not token.is_punct and not token.is_space
    ]
    if not tags:
        return np.nan
    counts = Counter(tags)
    return sum(counts[tag] for tag in CONTENT_POS) / len(tags)


def mean_concreteness(tokens: list[str], ratings: dict):
    values = [
        ratings[token.lower()]
        for token in tokens
        if token.isalpha() and token.lower() in ratings
    ]
    return float(np.mean(values)) if values else np.nan


def mean_aoa(doc, ratings: dict):
    values = [
        ratings[token.text.lower()]
        for token in doc
        if token.pos_ in CONTENT_POS
        and token.text.isalpha()
        and token.text.lower() in ratings
    ]
    return float(np.mean(values)) if values else np.nan


def rounded(value):
    return round(float(value), 3) if not np.isnan(value) else np.nan


def compute_lexical_metrics(text: str) -> dict:
    """Calculate the readability and lexical measures for one essay."""
    tokens, doc = parse_text(text)
    richness = LexicalRichness(text)
    richness_token_count = len(richness.wordlist)

    # shortest essay has 14 tokens, so shorter texts use their available length
    hdd = richness.hdd(draws=min(42, richness_token_count - 1))
    mattr = richness.mattr(window_size=min(20, richness_token_count))

    frequencies, ranks = load_freq_list()
    # surface forms are used here, matching the method described in the thesis
    concreteness = load_concreteness_dict(
        str(RESOURCE_DIR / "concreteness_en.csv")
    )
    aoa = load_aoa_dict(str(RESOURCE_DIR / "AoA_51715_words.xlsx"))

    return {
        "total_tokens": len(tokens),
        "flesch_reading_ease": round(float(textstat.flesch_reading_ease(text)), 3),
        "flesch_kincaid_grade": round(float(textstat.flesch_kincaid_grade(text)), 3),
        "gunning_fog": round(float(textstat.gunning_fog(text)), 3),
        "ari": round(float(textstat.automated_readability_index(text)), 3),
        "mtld": rounded(richness.mtld()),
        "mattr": rounded(mattr),
        "hdd": rounded(hdd),
        "lexical_density": rounded(lexical_density(doc)),
        "high_freq_coverage": rounded(high_freq_coverage(tokens, ranks)),
        "low_frequency_ratio": rounded(low_frequency_ratio(tokens, ranks)),
        "mean_log_frequency": rounded(average_log_frequency(tokens, frequencies)),
        "avg_word_concreteness": rounded(mean_concreteness(tokens, concreteness)),
        "mean_aoa": rounded(mean_aoa(doc, aoa)),
    }
