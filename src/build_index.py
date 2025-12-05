from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def main():
    corpus_path = PROC_DIR / "docs_corpus.csv"
    df = pd.read_csv(corpus_path)

    print(f"Loaded {len(df)} document chunks for the knowledge base")

    texts = df["text"].astype(str).tolist()
    doc_ids = df["doc_id"].tolist()

    # 1) Load sentence embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loaded embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # 2) Encode all documents
    print("Start encoding documents...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print("Encoding finished, shape:", embeddings.shape)

    # 3) Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("FAISS index built, number of vectors:", index.ntotal)

    # 4) Save index, embeddings and document metadata
    faiss_path = INDEX_DIR / "corpus.index"
    faiss.write_index(index, str(faiss_path))

    emb_path = INDEX_DIR / "corpus_embeddings.npy"
    np.save(emb_path, embeddings)

    meta_path = INDEX_DIR / "corpus_meta.csv"
    df.to_csv(meta_path, index=False, encoding="utf-8")

    # Save embedding model name for later loading
    model_name_path = INDEX_DIR / "embedding_model.txt"
    model_name_path.write_text(model_name, encoding="utf-8")

    print(f"Index saved to: {faiss_path}")
    print(f"Embeddings saved to: {emb_path}")
    print(f"Document metadata saved to: {meta_path}")
    print(f"Embedding model name saved to: {model_name_path}")


if __name__ == "__main__":
    main()