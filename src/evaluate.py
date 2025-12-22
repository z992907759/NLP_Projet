import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# CONFIGURATION ET IMPORTS

# On a besoin d'importer les briques technologiques depuis main.py
# (FAISS, BM25, Fusion, Reranking)
try:
    sys.path.append("src")
    from main import (
        load_resources, 
        retrieve_faiss, 
        retrieve_bm25, 
        reciprocal_rank_fusion, 
        rerank_contexts
    )
except ImportError:
    # Fallback si lancé depuis le dossier src directement
    from main import (
        load_resources, 
        retrieve_faiss, 
        retrieve_bm25, 
        reciprocal_rank_fusion, 
        rerank_contexts
    )

# Chemins des fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "golden_dataset.json"
OUTPUT_FILENAME = "evaluation_results_v4_hybrid.csv"

# Seuil de pertinence du Cross-Encoder.
# Contrairement à la distance Cosinus (0 à 1), le Cross-Encoder sort des "logits".
# Une valeur < 0 signifie généralement que le document n'est pas pertinent.
THRESHOLD_STRICT = 0.0 


# PIPELINE D'ÉVALUATION (SIMULATION V4)

def retrieve_eval_pipeline(query, df, index, embed_model, bm25, reranker, top_k=5):
    # 1. Branche Dense (Sémantique - FAISS)
    # On récupère large (top 15) pour ne rien rater
    res_faiss = retrieve_faiss(query, df, index, embed_model, top_k=15)
    
    # 2. Branche Sparse (Mots-clés - BM25)
    # On récupère large aussi pour attraper les acronymes/codes
    res_bm25 = retrieve_bm25(query, df, bm25, top_k=15)
    
    # 3. Fusion (RRF)
    # On combine les deux listes mathématiquement
    fused_docs = reciprocal_rank_fusion(res_faiss, res_bm25)
    
    # 4. Reranking Final (Cross-Encoder)
    # Le modèle expert relit les candidats et ne garde que le Top K
    final_docs = rerank_contexts(query, fused_docs, reranker, top_k=top_k)
    
    return final_docs


# CALCUL DES MÉTRIQUES

def calculate_metrics(retrieved_docs, expected_ids):

    if not expected_ids:
        return False, 0.0 # Cas particulier des questions pièges
    
    found_ids = [d.get('doc_id') for d in retrieved_docs]
    
    # Succès = Au moins UN des documents attendus a été trouvé
    success = any(e_id in found_ids for e_id in expected_ids)
    
    # On calcule la moyenne des scores des bons documents (pour info)
    relevant_scores = [d['score'] for d in retrieved_docs if d.get('doc_id') in expected_ids]
    avg_score = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
    
    return success, avg_score


# MAIN (BOUCLE DE TEST)

def main():
    print(f"=== ÉVALUATION SCIENTIFIQUE : MODE HYBRIDE V4 ===")
    
    # 1. Vérification du Dataset
    if not DATASET_PATH.exists():
        print(f"[ERREUR] Le fichier {DATASET_PATH} n'existe pas.")
        print("Veuillez créer 'data/golden_dataset.json' avant de lancer l'évaluation.")
        return
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] {len(dataset)} questions chargées pour le test.")

    # 2. Chargement du Moteur Complet (5 ressources)
    print("[INFO] Chargement du moteur (FAISS, BM25, Cross-Encoder)...")
    # Attention : load_resources renvoie maintenant 5 objets
    df, index, embed_model, reranker, bm25 = load_resources()
    
    results = []
    correct_retrieval_count = 0
    trap_success_count = 0
    total_scorable = 0 # Nombre de questions "normales" (pas des pièges)
    
    print("\n--- Exécution du Benchmark ---")
    
    for item in tqdm(dataset):
        query = item["question"]
        q_type = item["type"]
        expected_ids = item.get("expected_doc_ids", [])
        
        # --- APPEL DU PIPELINE V4 ---
        retrieved_contexts = retrieve_eval_pipeline(query, df, index, embed_model, bm25, reranker, top_k=5)
        
        # --- ANALYSE DES RÉSULTATS ---
        
        if q_type == "trap":
            # CAS 1 : Question Piège (Trap)
            # Le but est que le modèle ne trouve rien ou ait un score très bas.
            if not retrieved_contexts:
                max_score = -10.0 # Score arbitraire très bas
            else:
                max_score = max([c['score'] for c in retrieved_contexts])
            
            # Succès si le score max est SOUS le seuil strict
            is_success = max_score < THRESHOLD_STRICT
            
            if is_success:
                trap_success_count += 1
            
            results.append({
                "id": item["id"],
                "type": "TRAP",
                "success": is_success,
                "info": f"Max Score: {max_score:.2f} (Seuil < {THRESHOLD_STRICT})"
            })
            
        else:
            # CAS 2 : Question Standard
            total_scorable += 1
            is_success, avg_conf = calculate_metrics(retrieved_contexts, expected_ids)
            
            if is_success:
                correct_retrieval_count += 1
            
            found_ids = [c.get('doc_id') for c in retrieved_contexts]
            results.append({
                "id": item["id"],
                "type": q_type,
                "success": is_success,
                "info": f"Attendu: {expected_ids} | Trouvé: {found_ids} | Score: {avg_conf:.2f}"
            })

    # 3. Génération du Rapport Final
    accuracy = (correct_retrieval_count / total_scorable * 100) if total_scorable > 0 else 0
    
    print("\n" + "="*50)
    print(f"    RAPPORT DE PERFORMANCE (V4 HYBRID)    ")
    print("="*50)
    print(f" PRÉCISION DU RETRIEVAL : {accuracy:.2f}%")
    print("   (Capacité à trouver le bon document dans le top 5)")
    print("-" * 50)
    
    nb_traps = len(dataset) - total_scorable
    if nb_traps > 0:
        trap_acc = (trap_success_count / nb_traps) * 100
        print(f"🛡️  FILTRAGE DES PIÈGES    : {trap_acc:.2f}%")
        print("   (Capacité à rejeter les questions hors-sujet)")
    
    print("-" * 50)
    
    # Sauvegarde CSV
    out_path = BASE_DIR / "data" / OUTPUT_FILENAME
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"[INFO] Détails sauvegardés dans : {out_path}")

if __name__ == "__main__":
    main()