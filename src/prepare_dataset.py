import pandas as pd
from paths import ELLIPSE_ALL, RAW_DATA_DIR


def main() -> None:
    train = pd.read_csv(RAW_DATA_DIR / "ELLIPSE_Final_github_train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "ELLIPSE_Final_github_test.csv")

    # the original split is not needed because no LLM is fine-tuned
    essays = pd.concat([train, test], ignore_index=True)

    # one essay per line makes later updates easier
    ELLIPSE_ALL.parent.mkdir(parents=True, exist_ok=True)
    essays.to_json(ELLIPSE_ALL, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(essays)} essays to {ELLIPSE_ALL}.")


if __name__ == "__main__":
    main()
