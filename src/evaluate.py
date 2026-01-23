import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi



# 1. CONFIGURATION ET CHEMINS

# Détection automatique de la racine du projet
BASE_DIR = Path.cwd()
while not (BASE_DIR / "data").exists():
    if BASE_DIR == BASE_DIR.parent:
        raise FileNotFoundError("Dossier 'data' introuvable.")
    BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
DATASET_PATH = DATA_DIR / "golden_dataset.json"

# Seuils de décision
THRESHOLD_STRICT = -3      # Pour V4 (Logits du Reranker)
THRESHOLD_SIMPLE = 0.3     # Pour V1 (Similarité Cosinus FAISS)


# 2. CHARGEMENT DES RESSOURCES

def load_resources():
    print("--- Chargement des ressources RAG ---")
    
    # 1. Corpus (Textes)
    corpus_path = PROC_DIR / "docs_corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {corpus_path}")
    df = pd.read_csv(corpus_path)
    
    # 2. Index FAISS (Vecteurs)
    faiss_path = INDEX_DIR / "corpus.index"
    index = faiss.read_index(str(faiss_path))

    # 3. Modèle d'Embedding (Traducteur Texte -> Vecteur)
    model_name_path = INDEX_DIR / "embedding_model.txt"
    model_name = model_name_path.read_text(encoding="utf-8").strip()
    embed_model = SentenceTransformer(model_name)
    
    # 4. Reranker (Juge de pertinence)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 5. Index BM25 (Mots-clés)
    tokenized_corpus = [str(doc).lower().split(" ") for doc in df['text']]
    bm25 = BM25Okapi(tokenized_corpus)

    return df, index, embed_model, reranker, bm25



# 3. BRIQUES DE RECHERCHE (ATOMIQUES)

def retrieve_faiss(query, df, index, embed_model, top_k=10):
    """Recherche Vectorielle (Sémantique)"""
    query_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_emb, top_k)
    contexts = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(df): continue
        row = df.iloc[idx]
        contexts.append({
            "doc_id": row.get("doc_id", idx),
            "score": float(score),
            "text": str(row["text"])
        })
    return contexts

def retrieve_bm25(query, df, bm25, top_k=10):
    """Recherche Lexicale (Mots-clés exacts)"""
    tokenized_query = query.lower().split(" ")
    top_docs = bm25.get_top_n(tokenized_query, df['text'].tolist(), n=top_k)
    contexts = []
    for text in top_docs:
        # Note : Retrouver la ligne via le texte est lent mais ok pour l'évaluation
        matches = df[df['text'] == text]
        if not matches.empty:
            row = matches.iloc[0]
            contexts.append({
                "doc_id": row.get("doc_id"),
                "score": 0.0, 
                "text": text
            })
    return contexts

def reciprocal_rank_fusion(list_a, list_b, k=60):
    """Fusionne les résultats FAISS et BM25"""
    scores_map = {}
    def add_to_map(results_list):
        for rank, doc in enumerate(results_list):
            key = doc['text'] 
            if key not in scores_map:
                scores_map[key] = {"doc": doc, "score": 0.0}
            scores_map[key]["score"] += 1 / (k + rank + 1)
    add_to_map(list_a)
    add_to_map(list_b)
    fused_sorted = sorted(scores_map.values(), key=lambda x: x['score'], reverse=True)
    return [item['doc'] for item in fused_sorted]

def rerank_contexts(query, contexts, reranker, top_k=5):
    """Réordonne les résultats avec le Cross-Encoder"""
    if not contexts: return []
    pairs = [[query, doc['text']] for doc in contexts]
    scores = reranker.predict(pairs)
    for i, doc in enumerate(contexts):
        doc['score'] = float(scores[i])
    return sorted(contexts, key=lambda x: x['score'], reverse=True)[:top_k]



# 4. PIPELINES À COMPARER

def pipeline_v1_simple(query, df, index, embed_model, top_k=5):
    """Pipeline V1 : Recherche FAISS simple"""
    return retrieve_faiss(query, df, index, embed_model, top_k)

def pipeline_v4_hybrid(query, df, index, embed_model, bm25, reranker, top_k=5):
    """Pipeline V4 : Hybride (FAISS+BM25) -> Fusion -> Reranking"""
    res_faiss = retrieve_faiss(query, df, index, embed_model, top_k=15)
    res_bm25 = retrieve_bm25(query, df, bm25, top_k=15)
    fused_docs = reciprocal_rank_fusion(res_faiss, res_bm25)
    return rerank_contexts(query, fused_docs, reranker, top_k=top_k)



# 5. CALCUL DES MÉTRIQUES

def calculate_success(retrieved_docs, expected_ids):
    """Renvoie True si au moins un ID de document attendu est trouvé"""
    if not expected_ids or not retrieved_docs:
        return False
    found_ids = [d.get('doc_id') for d in retrieved_docs]
    return any(e_id in found_ids for e_id in expected_ids)



# 6. EXÉCUTION DU BENCHMARK COMPARATIF

print(f"\n=== BENCHMARK AVANCÉ : V1 (Simple) vs V4 (Hybride) ===")

if not DATASET_PATH.exists():
    print(f"[ERREUR] Dataset introuvable : {DATASET_PATH}")
else:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] {len(dataset)} questions chargées.")

    # Chargement du moteur
    df, index, embed_model, reranker, bm25 = load_resources()

    # Structure de données pour stocker les résultats par catégorie
    # On sépare Direct et Synthesis pour voir l'impact du seuil
    results = {
        "v1": {
            "direct":    {"total": 0, "found_raw": 0, "found_prod": 0},
            "synthesis": {"total": 0, "found_raw": 0, "found_prod": 0},
            "trap":      {"total": 0, "caught": 0}
        },
        "v4": {
            "direct":    {"total": 0, "found_raw": 0, "found_prod": 0},
            "synthesis": {"total": 0, "found_raw": 0, "found_prod": 0},
            "trap":      {"total": 0, "caught": 0}
        }
    }

    print("\n--- Lancement des tests ---")

    for item in tqdm(dataset):
        query = item["question"]
        # Par défaut 'direct' si le type n'est pas spécifié dans le JSON
        q_type = item.get("type", "direct") 
        expected_ids = item.get("expected_doc_ids", [])

        # --- 1. EXÉCUTION DES PIPELINES ---
        docs_v1 = pipeline_v1_simple(query, df, index, embed_model, top_k=5)
        docs_v4 = pipeline_v4_hybrid(query, df, index, embed_model, bm25, reranker, top_k=5)

        # Récupération des meilleurs scores (ou valeur très basse si vide)
        score_v1 = docs_v1[0]['score'] if docs_v1 else 0.0
        score_v4 = docs_v4[0]['score'] if docs_v4 else -999.0

        # --- 2. ÉVALUATION V1 ---
        if q_type == "trap":
            results["v1"]["trap"]["total"] += 1
            # Succès si vide OU score < seuil
            if not docs_v1 or score_v1 < THRESHOLD_SIMPLE:
                results["v1"]["trap"]["caught"] += 1
        else:
            cat = q_type if q_type in ["direct", "synthesis"] else "direct"
            results["v1"][cat]["total"] += 1
            
            # A. Raw Retrieval (Est-ce que le doc est là ?)
            if calculate_success(docs_v1, expected_ids):
                results["v1"][cat]["found_raw"] += 1
                # B. Production (Est-ce que le doc est là ET score suffisant ?)
                if score_v1 >= THRESHOLD_SIMPLE:
                    results["v1"][cat]["found_prod"] += 1

        # --- 3. ÉVALUATION V4 ---
        if q_type == "trap":
            results["v4"]["trap"]["total"] += 1
            if not docs_v4 or score_v4 < THRESHOLD_STRICT:
                results["v4"]["trap"]["caught"] += 1
        else:
            cat = q_type if q_type in ["direct", "synthesis"] else "direct"
            results["v4"][cat]["total"] += 1
            
            if calculate_success(docs_v4, expected_ids):
                results["v4"][cat]["found_raw"] += 1
                if score_v4 >= THRESHOLD_STRICT:
                    results["v4"][cat]["found_prod"] += 1




# 7. RAPPORT FINAL (Mis à jour pour la lisibilité)

print("\n" + "="*85)
print(f"                            RAPPORT DE PERFORMANCE : V1 vs V4       ")
print("="*85)

def calculate_global_metrics(data_version):
    # Aggrégation Recherche (Direct + Synthesis)
    total_search = data_version["direct"]["total"] + data_version["synthesis"]["total"]
    found_raw = data_version["direct"]["found_raw"] + data_version["synthesis"]["found_raw"]
    found_prod = data_version["direct"]["found_prod"] + data_version["synthesis"]["found_prod"]

    acc_raw = (found_raw / total_search * 100) if total_search > 0 else 0.0
    acc_prod = (found_prod / total_search * 100) if total_search > 0 else 0.0

    # Aggrégation Pièges (Trap)
    total_traps = data_version["trap"]["total"]
    caught = data_version["trap"]["caught"]
    acc_trap = (caught / total_traps * 100) if total_traps > 0 else 0.0

    return acc_raw, acc_prod, acc_trap

v1_raw, v1_prod, v1_trap = calculate_global_metrics(results["v1"])
v4_raw, v4_prod, v4_trap = calculate_global_metrics(results["v4"])

# Fonction pour formater le gain avec une flèche
def fmt_gain(v4, v1):
    diff = v4 - v1
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}%"

summary_df = pd.DataFrame({
    "Ce que l'on mesure": [
        "Précision des documents sélectionnés", 
        "Succès Visible après filtrage", 
        "Questions pièges évitées"
    ],
    "V1 (Simple)": [
        f"{v1_raw:.1f}%", 
        f"{v1_prod:.1f}%", 
        f"{v1_trap:.1f}%"
    ],
    "V4 (Hybride)": [
        f"{v4_raw:.1f}%", 
        f"{v4_prod:.1f}%", 
        f"{v4_trap:.1f}%"
    ],
    "Évolution": [
        fmt_gain(v4_raw, v1_raw), 
        fmt_gain(v4_prod, v1_prod), 
        fmt_gain(v4_trap, v1_trap)
    ]
})

# Alignement à gauche pour la première colonne pour la lisibilité
print(summary_df.to_string(index=False, col_space=15, justify="center"))