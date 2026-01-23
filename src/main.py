# IMPORTS
from pathlib import Path
import textwrap
import warnings

# Bibliothèques de calcul et data
import faiss
import numpy as np
import pandas as pd
import torch

# Bibliothèques NLP (Sémantique & Reranking)
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# Bibliothèque NLP (Mots-clés)
from rank_bm25 import BM25Okapi 

import sys
from pathlib import Path

# CONFIGURATION GLOBALE

# Le modèle LLM
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Seuils de décision
THRESHOLD_SIMPLE = 0.35   # Pour FAISS (0 à 1).
THRESHOLD_RERANK = -4.00   # Pour Cross-Encoder (Logits -10 à +10). < 0 signifie Non pertinent.

print(f"[INIT] Loading LLM model: {MODEL_NAME}...")

# Détection automatique du matériel (GPU ou CPU)
if torch.cuda.is_available() or torch.backends.mps.is_available():
    llm_dtype = torch.bfloat16 # Mode rapide et léger pour GPU
    print("[INIT] Mode: GPU Acceleration (bfloat16)")
else:
    llm_dtype = torch.float32  # Mode standard pour CPU
    print("[INIT] Mode: CPU (float32)")


# CHARGEMENT DU LLM

# 1. Le Tokenizer : Convertit le texte en nombres
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Le Modèle
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=llm_dtype,
    device_map="auto", # Gestion automatique de la mémoire (VRAM/RAM)
)


# FONCTIONS UTILITAIRES LLM

def call_llm(prompt: str) -> str:
    """
    Envoie le prompt au LLM et récupère la réponse générée.
    Gère la mise en forme du template de chat (System + User).
    """
    # Définition du rôle système strict pour limiter les hallucinations
    messages = [
        {"role": "system", "content": "You are a precise research assistant. Answer using ONLY the provided contexts. If the answer is not in the contexts, say you don't know."},
        {"role": "user", "content": prompt},
    ]
    
    # Transformation des messages en tokens
    if hasattr(tokenizer, "apply_chat_template"):
        model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    else:
        # Fallback manuel si la fonction n'existe pas
        chat_text = f"System: Helper.\nUser:\n{prompt}\n\nAssistant:"
        model_inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=4096)

    # Si model_inputs est juste un Tensor, on le met dans un dictionnaire
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs}
    
    # Envoi des données sur le bon périphérique (GPU ou CPU)
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)}

    # Génération
    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs, 
            max_new_tokens=512, 
            do_sample=False, # Déterministe, pas de créativité aléatoire
            pad_token_id=tokenizer.eos_token_id
        )

    # Décodage (Tokens vers Texte)
    generated_ids = output_ids[0][model_inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def call_llm_baseline(query: str) -> str:
    """Appelle le LLM sans contexte (Mémoire interne uniquement)."""
    messages = [{"role": "user", "content": query}]
    
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    
    # Correction par sécurité
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs}
        
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
    return tokenizer.decode(output_ids[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def build_prompt(query: str, contexts):
    """Construit le prompt final avec les documents récupérés."""
    context_str = "\n\n".join([f"[Context {i+1} | score={c.get('score', 0):.2f}]\n{c['text']}" for i, c in enumerate(contexts)])
    return f"CONTEXTS:\n{context_str}\n\nQUESTION:\n{query}\n\nAnswer based on contexts only."

def _fallback(query, info=""):
    """Gère le cas où aucun document pertinent n'est trouvé."""
    baseline = call_llm_baseline(query)
    return f"*** [RAG FAILED: {info}] Fallback to Baseline (Connaissances générales) ***\n\n{baseline}"


# GESTION DES RESSOURCES (Index, Corpus, Modèles)

sys.path.append("..")
BASE_DIR = Path("..")

PROC_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index"


def load_resources():
    print("--- Loading RAG Resources ---")
    
    # 1. Corpus Textuel (CSV)
    corpus_path = PROC_DIR / "docs_corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {corpus_path}. Lancez docs_to_corpus.py d'abord.")
    df = pd.read_csv(corpus_path)
    
    # 2. Index Vectoriel (FAISS)
    faiss_path = INDEX_DIR / "corpus.index"
    index = faiss.read_index(str(faiss_path))

    # 3. Modèle d'Embedding (Bi-Encoder)
    model_name_path = INDEX_DIR / "embedding_model.txt"
    model_name = model_name_path.read_text(encoding="utf-8").strip()
    embed_model = SentenceTransformer(model_name)
    
    # 4. Modèle de Reranking (Cross-Encoder)
    print("Loading Reranker (Cross-Encoder)...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 5. Index Mots-clés (BM25)
    print("Building BM25 Index (Sparse Retrieval)...")
    # Tokenisation simple par espace pour BM25
    tokenized_corpus = [str(doc).lower().split(" ") for doc in df['text']]
    bm25 = BM25Okapi(tokenized_corpus)

    return df, index, embed_model, reranker, bm25




# BRIQUES DE BASE (Retrieval Modules)

def retrieve_faiss(query: str, df, index, embed_model, top_k=10):
    """Recherche Vectorielle (Dense Retrieval) via FAISS."""
    query_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_emb, top_k)
    contexts = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(df): continue
        row = df.iloc[idx]
        contexts.append({
            "doc_id": row.get("doc_id", idx),
            "score": float(score),
            "text": str(row["text"]),
            "source": "FAISS"
        })
    return contexts

def retrieve_bm25(query: str, df, bm25, top_k=10):
    """Recherche Lexicale (Sparse Retrieval) via BM25."""
    tokenized_query = query.lower().split(" ")
    top_docs = bm25.get_top_n(tokenized_query, df['text'].tolist(), n=top_k)
    
    contexts = []
    for text in top_docs:
        contexts.append({
            "doc_id": "BM25_Match",
            "score": 0.0, # BM25 score non normalisé, ignoré ici au profit du rang
            "text": text,
            "source": "BM25"
        })
    return contexts

def reciprocal_rank_fusion(list_a, list_b, k=60):
    """
    Fusionne deux listes de résultats (ex: FAISS + BM25) via RRF.
    Score = 1 / (k + rang). Favorise les documents présents dans les deux listes.
    """
    scores_map = {}
    
    def add_to_map(results_list):
        for rank, doc in enumerate(results_list):
            key = doc['text'] # Clé de dédoublage
            if key not in scores_map:
                scores_map[key] = {"doc": doc, "score": 0.0}
            scores_map[key]["score"] += 1 / (k + rank + 1)
            
    add_to_map(list_a)
    add_to_map(list_b)
    
    fused_sorted = sorted(scores_map.values(), key=lambda x: x['score'], reverse=True)
    return [item['doc'] for item in fused_sorted]

def rerank_contexts(query, contexts, reranker, top_k=5):
    """
    Réordonne les candidats en utilisant un Cross-Encoder.
    C'est le 'Juge' qui lit la question et le document ensemble.
    """
    if not contexts: return []
    pairs = [[query, doc['text']] for doc in contexts]
    scores = reranker.predict(pairs)
    for i, doc in enumerate(contexts):
        doc['score'] = float(scores[i])
    # Tri décroissant selon le nouveau score de pertinence
    return sorted(contexts, key=lambda x: x['score'], reverse=True)[:top_k]



# FONCTION D'AFFICHAGE DES SOURCES

def display_sources(contexts):
    """
    Affiche un bloc visuel avec les sources utilisées pour générer la réponse.
    Critère d'acceptation : Nom fichier, ID Chunk, Score.
    """
    if not contexts:
        return

    print("\n" + "   " + "─"*50)
    print("   📚 SOURCES UTILISÉES (Preuves)")
    print("   " + "─"*50)
    
    for i, doc in enumerate(contexts):
        # Récupération des métadonnées
        source = doc.get('source', 'Document inconnu')
        doc_id = doc.get('doc_id', '?')
        score = doc.get('score', 0.0)
        
        # Petit extrait du texte pour le contexte (optionnel mais classe)
        snippet = doc['text'][:85].replace("\n", " ") + "..."
        
        # Affichage formaté
        # On met le score en gras ou en évidence visuelle simple
        print(f"   {i+1}. 📄 {source} | Chunk #{doc_id}")
        print(f"      🎯 Confiance : {score:.4f}")
        print(f"      📝 Extrait : \"{snippet}\"")
        print("   " + "-"*20)
    print("\n")



# PIPELINES RAG (Versions V1 à V4)

# ====== V1: SIMPLE RAG (Baseline) ======
def rag_v1(query, df, index, embed_model, top_k=5):
    docs = retrieve_faiss(query, df, index, embed_model, top_k)
    display_sources(docs)
    if not docs or docs[0]['score'] < THRESHOLD_SIMPLE:
        return _fallback(query, "Low FAISS Score")
    return call_llm(build_prompt(query, docs))



# ====== V2: MULTI-QUERY (Expansion de requête) ======
def rag_v2(query, df, index, embed_model, top_k=5):
    print("   [V2] Generating variations...")
    variations = [query]
    try:
        gen = call_llm_baseline(f"Generate 2 alternative questions for: {query}")
        variations += [line for line in gen.split('\n') if "?" in line][:2]
    except: pass
    
    candidates = {}
    for q in variations:
        for doc in retrieve_faiss(q, df, index, embed_model, top_k=3):
            candidates[doc['text']] = doc
            
    final_docs = list(candidates.values())[:top_k*2]
    if not final_docs: return _fallback(query)
    return call_llm(build_prompt(query, final_docs[:top_k]))






# ====== V3: RERANKING (Filtrage Avancé) ======
def rag_v3(query, df, index, embed_model, reranker, top_k=5):
    variations = [query]
    
    # Récupération large (Recall)
    candidates = {}
    for q in variations:
        for doc in retrieve_faiss(q, df, index, embed_model, top_k=15):
            candidates[doc['text']] = doc
    
    unique_candidates = list(candidates.values())
    
    # Filtrage précis (Precision)
    final_docs = rerank_contexts(query, unique_candidates, reranker, top_k=top_k)
    
    if final_docs:
        print(f"   [V3] Top Score après Rerank: {final_docs[0]['score']:.2f}")

    if not final_docs or final_docs[0]['score'] < THRESHOLD_RERANK:
        return _fallback(query, "Low Rerank Score (< 0.0)")
        
    return call_llm(build_prompt(query, final_docs))




# ====== V4: HYBRID ======
def rag_v4_hybrid(query, df, index, embed_model, bm25, reranker, top_k=5):
    print("=== [V4] Pipeline: Multi-Query -> Hybrid (FAISS+BM25) -> RRF -> Reranking ===")
    
    # 1. Expansion
    variations = [query]
    candidates = {}
    
    for q in variations:
        # A. Dense Retrieval
        res_faiss = retrieve_faiss(q, df, index, embed_model, top_k=10)
        # B. Sparse Retrieval
        res_bm25 = retrieve_bm25(q, df, bm25, top_k=10)
        # C. Fusion
        fused = reciprocal_rank_fusion(res_faiss, res_bm25)
        
        for doc in fused[:10]:
            candidates[doc['text']] = doc
            
    unique_candidates = list(candidates.values())
    print(f"   [V4] {len(unique_candidates)} candidats uniques identifiés.")

    # 2. Reranking
    final_docs = rerank_contexts(query, unique_candidates, reranker, top_k=top_k)
    
    if not final_docs: return _fallback(query)
    
    print(f"   [V4] Meilleur Score Final: {final_docs[0]['score']:.2f}")
    
    display_sources(final_docs)

    if final_docs[0]['score'] < THRESHOLD_RERANK:
         return _fallback(query, "Irrelevant Context")
    
    return call_llm(build_prompt(query, final_docs))



# DEMO with "For which population groups is assessment of total CVD risk recommended?"

def main():
    df, index, embed_model, reranker, bm25 = load_resources()

    print("\n" + "="*60)
    print("      RAG SYSTEM : DEMO (Comparative Mode)      ")
    print("      Comparez V1 (Simple) vs V4 (Hybride Avancée)")
    print("="*60)
    
    while True:
        try:
            query = input("\nVotre question (ou 'q' pour quitter) : ").strip()
        except EOFError:
            break
            
        if query.lower() in {"q", "quit", "exit"}:
            print("Arrêt du système. Au revoir !")
            break
        
        if not query:
            continue

        print("\n" + "-"*30 + " V1. SIMPLE (FAISS) " + "-"*30)
        print(rag_v1(query, df, index, embed_model))

        print("\n" + "-"*30 + " V4. HYBRID ULTIMATE (BM25+FAISS+RERANK) " + "-"*30)
        print(rag_v4_hybrid(query, df, index, embed_model, bm25, reranker))
        
        print("\n" + "="*60)

main()