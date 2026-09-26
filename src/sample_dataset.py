import pandas as pd
from paths import ELLIPSE_ALL, ELLIPSE_SAMPLE

TARGET_TOTAL = 1200
RARE_THRESHOLD = 160
RANDOM_STATE = 42


def main() -> None:
    essays = pd.read_json(ELLIPSE_ALL, lines=True)
    counts = essays["Overall"].value_counts().sort_index()
    rare_scores = counts[counts <= RARE_THRESHOLD].index
    common_scores = counts[counts > RARE_THRESHOLD].index

    # keep all rare scores, then divide the remaining places
    remaining = TARGET_TOTAL - counts.loc[rare_scores].sum()
    base, extra = divmod(remaining, len(common_scores))
    groups = [
        essays[essays["Overall"] == score].sample(
            n=min(counts[score], base + int(index < extra)),
            # same sample whenever the script is rerun
            random_state=RANDOM_STATE,
        )
        for index, score in enumerate(common_scores)
    ]
    groups.append(essays[essays["Overall"].isin(rare_scores)])

    # shuffle after combining the score groups
    sample = pd.concat(groups, ignore_index=True)
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    ELLIPSE_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_json(ELLIPSE_SAMPLE, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(sample)} essays to {ELLIPSE_SAMPLE}.")


if __name__ == "__main__":
    main()
