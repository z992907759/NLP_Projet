from pathlib import Path
import json
import pandas as pd


RAW_DIR = Path("../data/raw")
CLEAN_DIR = Path("../data/processed")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset1(path: Path):
    """agriculture_dataset1.json(JSONL) -> list[dict(question, answer, source)]"""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("input") or "").strip()
            a = (obj.get("response") or "").strip()
            if q and a:
                rows.append(
                    {
                        "question": q,
                        "answer": a,
                        "source": "dataset1",
                    }
                )
    return rows


def load_dataset2(path: Path):
    """agriculture_dataset2.json(JSONL) -> list[dict(question, answer, source)]"""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations") or []
            user_msgs = [m["content"] for m in conv if m.get("role") == "user"]
            assistant_msgs = [m["content"] for m in conv if m.get("role") == "assistant"]

            if not user_msgs or not assistant_msgs:
                continue

            q = user_msgs[0].strip()
            a = assistant_msgs[-1].strip()
            if q and a:
                rows.append(
                    {
                        "question": q,
                        "answer": a,
                        "source": "dataset2",
                    }
                )
    return rows


def load_csv(path: Path):
    """agriculture_qa.csv -> list[dict(question, answer, source)]"""
    df = pd.read_csv(path)

    q_col = "question"
    a_col = "answers"

    df[q_col] = df[q_col].astype(str).str.strip()
    df[a_col] = df[a_col].astype(str).str.strip()

    df = df[(df[q_col] != "") & (df[a_col] != "")]
    df["source"] = "csv"

    df = df.rename(columns={q_col: "question", a_col: "answer"})
    return df[["question", "answer", "source"]].to_dict(orient="records")


def main():
    rows = []
    # 1) Load three data sources
    ds1_path = RAW_DIR / "agriculture_dataset1.json"
    ds2_path = RAW_DIR / "agriculture_dataset2.json"
    csv_path = RAW_DIR / "agriculture_qa.csv"

    if ds1_path.exists():
        rows.extend(load_dataset1(ds1_path))
    if ds2_path.exists():
        rows.extend(load_dataset2(ds2_path))
    if csv_path.exists():
        rows.extend(load_csv(csv_path))

    # 2) Merge into DataFrame
    df = pd.DataFrame(rows)

    if df.empty:
        print("[ERROR] No data loaded. Check that raw datasets exist in data/raw and filenames are correct.")
        print(f"Tried paths: {ds1_path}, {ds2_path}, {csv_path}")
        return

    # 3) Basic cleaning: remove nulls, extra whitespace, and duplicates
    df["question"] = (
        df["question"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["answer"] = (
        df["answer"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )

    # Remove Q&A pairs that are too short
    df = df[(df["question"].str.len() > 1) & (df["answer"].str.len() > 3)]

    # Remove duplicate Q&A pairs
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

    # Add incremental id column
    df.insert(0, "id", df.index + 1)

    # 4) Save cleaned dataset
    out_path = CLEAN_DIR / "agriculture_qa_merged_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()