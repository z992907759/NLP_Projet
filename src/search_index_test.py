from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index"


def load_resources():
    # 1) Load metadata
    corpus_path = PROC_DIR / "docs_corpus.csv"
    df = pd.read_csv(corpus_path)

    # 2) Load FAISS index
    faiss_path = INDEX_DIR / "corpus.index"
    index = faiss.read_index(str(faiss_path))

    # 3) Load embedding model
    model_name_path = INDEX_DIR / "embedding_model.txt"
    model_name = model_name_path.read_text(encoding="utf-8").strip()
    print(f"Loaded embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    return df, index, model


def search(query, df, index, model, top_k=5):
    # 1) Encode query
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 2) Search in FAISS
    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        text = str(row["text"])
        doc_id = row.get("doc_id", idx)

        results.append(
            {
                "doc_id": doc_id,
                "score": float(score),
                "text": text,
            }
        )
    return results


def main():
    df, index, model = load_resources()

    print("=== Simple retrieval demo: enter a query and press Enter ===")
    print("Type q or quit to exit.")

    while True:
        query = input("\nPlease enter your query: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            print("Bye ~")
            break

        results = search(query, df, index, model, top_k=5)

        print("\nTop-5 retrieval results:")
        for i, r in enumerate(results, start=1):
            # Show only first 200 characters to avoid long output
            preview = r["text"][:200].replace("\n", " ")
            print(f"[{i}] doc_id={r['doc_id']}  score={r['score']:.4f}")
            print(f"    {preview}...")


if __name__ == "__main__":
    main()