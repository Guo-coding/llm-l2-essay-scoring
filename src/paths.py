"""Shared repository paths used by the research scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# shared locations used throughout the scripts
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESOURCE_DIR = DATA_DIR / "resources"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# files produced during data preparation and scoring
ELLIPSE_ALL = INTERIM_DATA_DIR / "ellipse_all.jsonl"
ELLIPSE_SAMPLE = PROCESSED_DATA_DIR / "ellipse_sampled_1200.jsonl"
ELLIPSE_RESULTS = PROCESSED_DATA_DIR / "ellipse_results.jsonl"


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    # leave external paths unchanged
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
