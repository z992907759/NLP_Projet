from pathlib import Path
import sys

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Project root directory
sys.path.append("..") 

# CONFIGURATION DES CHEMINS
BASE_DIR = Path("..")

# Dossier où se trouve le CSV (Données traitées)
PROC_DIR = BASE_DIR / "data" / "processed"

# Dossier où on va sauvegarder l'index
INDEX_DIR = BASE_DIR / "data" / "index"

# Création du dossier s'il n'existe pas
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def main():
    corpus_path = PROC_DIR / "docs_corpus.csv"
    df = pd.read_csv(corpus_path)

    print(f"Loaded {len(df)} document chunks for the knowledge base")

    texts = df["text"].astype(str).tolist()
    doc_ids = df["doc_id"].tolist()

    # 1. Chargement du modèle de sentence embedding
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loaded embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # 2. Encoder tous les documents
    print("Start encoding documents...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print("Encoding finished, shape:", embeddings.shape)

    # 3. Construction de l'index FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("FAISS index built, number of vectors:", index.ntotal)

    # 4. Sauvegarde de l'index, des embeddings et des métadonnées des documents
    faiss_path = INDEX_DIR / "corpus.index"
    faiss.write_index(index, str(faiss_path))

    emb_path = INDEX_DIR / "corpus_embeddings.npy"
    np.save(emb_path, embeddings)

    meta_path = INDEX_DIR / "corpus_meta.csv"
    df.to_csv(meta_path, index=False, encoding="utf-8")

    # Sauvagardes du nom du modèle d'embedding pour pouvoir le charger plus tard
    model_name_path = INDEX_DIR / "embedding_model.txt"
    model_name_path.write_text(model_name, encoding="utf-8")

    print(f"Index saved to: {faiss_path}")
    print(f"Embeddings saved to: {emb_path}")
    print(f"Document metadata saved to: {meta_path}")
    print(f"Embedding model name saved to: {model_name_path}")


if __name__ == "__main__":
    main()