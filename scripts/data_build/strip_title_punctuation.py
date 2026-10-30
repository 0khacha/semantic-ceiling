"""
Build the canonical v2 dataset by removing trailing title punctuation.

PubMed/MEDLINE titles (the negatives) end with a period, RWDB-sourced titles
(the positives) do not. Fine-tuned transformers learned this end-of-sequence
token instead of any content signal (thesis §5.8), so v2 strips trailing
periods, semicolons and spaces from every title: title.rstrip(" .;").
Abstracts and all other fields are left unchanged.

Input:  data/interim/cancer_pm_{dataset,train,val,test,holdout}.json   (v1)
Output: data/processed/cancer_pm_{dataset,train,val,test,holdout}.json (v2)

Applied to the archived v1 files (archive/leaky_data_v1/) this reproduces every
record of data/processed exactly (5,536 / 1,188 / 1,188 / 1,074 titles changed
in train / val / test / holdout).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, ROOT  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("dataset", "train", "val", "test", "holdout"):
        with open(INTERIM_DIR / f"cancer_pm_{name}.json", encoding="utf-8") as f:
            records = json.load(f)
        changed = 0
        for r in records:
            stripped = r["title"].rstrip(" .;")
            changed += stripped != r["title"]
            r["title"] = stripped
        with open(DATA_DIR / f"cancer_pm_{name}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"{name:<8} {changed:>5} / {len(records)} titles changed")


if __name__ == "__main__":
    main()
