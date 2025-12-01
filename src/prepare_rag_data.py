from pathlib import Path
import pandas as pd

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

def main():
    src_path = PROC_DIR / "agriculture_qa_merged_clean.csv"
    df = pd.read_csv(src_path)

    print(f"Total number of samples: {len(df)}")

    # 1) Build evaluation set: randomly sample 500 rows (or all if fewer than 500)
    eval_size = min(500, len(df))
    eval_df = df.sample(n=eval_size, random_state=42).reset_index(drop=True)
    eval_path = PROC_DIR / "qa_eval.csv"
    eval_df.to_csv(eval_path, index=False, encoding="utf-8")
    print(f"Saved evaluation set with {len(eval_df)} rows to {eval_path}")

    # 2) Build RAG corpus: use all unique answers as documents
    corpus_df = df[["answer"]].drop_duplicates().reset_index(drop=True)
    corpus_df.insert(0, "doc_id", corpus_df.index)
    corpus_df = corpus_df.rename(columns={"answer": "text"})
    corpus_path = PROC_DIR / "corpus.csv"
    corpus_df.to_csv(corpus_path, index=False, encoding="utf-8")
    print(f"Saved knowledge base corpus with {len(corpus_df)} documents to {corpus_path}")


if __name__ == "__main__":
    main()