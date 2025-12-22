import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys


# Mettre à True pour tester le Multi-Query, False pour la recherche simple
SIMILARITY_THRESHOLD = 0.5
USE_MULTI_QUERY = True 
OUTPUT_FILENAME = "evaluation_results_multiquery.csv" if USE_MULTI_QUERY else "evaluation_results_baseline.csv"


# On essaie d'importer les fonctions depuis main.py
try:
    from main import load_resources, retrieve, retrieve_multi_query
except ImportError:
    sys.path.append("src")
    from main import load_resources, retrieve, retrieve_multi_query

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "golden_dataset.json"

def calculate_metrics(retrieved_docs, expected_ids):
    """
    Vérifie si l'un des documents attendus est présent dans les résultats récupérés.
    """
    if not expected_ids:
        return False, 0.0 # Cas des questions pièges
    
    found_ids = [d.get('doc_id') for d in retrieved_docs]
    
    # Est-ce qu'on a trouvé au moins UN bon document ?
    success = any(e_id in found_ids for e_id in expected_ids)
    
    # Calcul du score moyen des bons documents trouvés
    relevant_scores = [d['score'] for d in retrieved_docs if d.get('doc_id') in expected_ids]
    avg_score = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
    
    return success, avg_score

def main():
    mode_str = "MULTI-QUERY" if USE_MULTI_QUERY else "SIMPLE RETRIEVAL"
    print(f"=== Démarrage de l'évaluation automatique : {mode_str} ===")
    
    # Chargement du Golden Dataset
    if not DATASET_PATH.exists():
        print(f"[ERREUR] Le fichier {DATASET_PATH} n'existe pas.")
        return
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] {len(dataset)} questions chargées depuis le Golden Dataset.")

    # Chargement du moteur RAG
    print("[INFO] Chargement des ressources RAG (Index, Modèle...)...")
    df, index, embed_model = load_resources()
    
    # Boucle d'évaluation
    results = []
    correct_retrieval_count = 0
    trap_success_count = 0
    
    total_scorable = 0 
    
    print("\n--- Lancement des tests ---")
    for item in tqdm(dataset):
        query = item["question"]
        q_type = item["type"]
        expected_ids = item.get("expected_doc_ids", [])
        
        if USE_MULTI_QUERY:
            # On utilise la fonction avancée
            retrieved_contexts = retrieve_multi_query(query, df, index, embed_model, top_k=5)
        else:
            # On utilise l'ancienne fonction simple
            retrieved_contexts = retrieve(query, df, index, embed_model, top_k=5)
        
        # Analyse des résultats
        if q_type == "trap":
            # Pour un piège, succès = aucun document trouvé OU scores très bas
            if not retrieved_contexts:
                max_score = 0
            else:
                max_score = max([c['score'] for c in retrieved_contexts])
            
            # On considère réussi si le score max est sous le seuil de pertinence (ex: 0.5)
            is_success = max_score < SIMILARITY_THRESHOLD
            if is_success:
                trap_success_count += 1
            
            results.append({
                "id": item["id"],
                "question": query,
                "type": "trap",
                "success": is_success,
                "info": f"Max Score: {max_score:.4f} (Seuil: {SIMILARITY_THRESHOLD})"
            })
            
        else:
            # Pour une question normale
            total_scorable += 1
            is_success, avg_conf = calculate_metrics(retrieved_contexts, expected_ids)
            
            if is_success:
                correct_retrieval_count += 1
            
            found_ids = [c.get('doc_id') for c in retrieved_contexts]
            results.append({
                "id": item["id"],
                "question": query,
                "type": q_type,
                "success": is_success,
                "info": f"Attendu: {expected_ids} | Trouvé: {found_ids}"
            })

    # Génération du Rapport
    accuracy = (correct_retrieval_count / total_scorable * 100) if total_scorable > 0 else 0
    
    print("\n" + "="*40)
    print(f"    RAPPORT D'ÉVALUATION ({mode_str})    ")
    print("="*40)
    print(f"Questions valides testées : {total_scorable}")
    print("-" * 40)
    print(f"✅ PRÉCISION DU RETRIEVAL : {accuracy:.2f}%")
    print("-" * 40)
    
    nb_traps = len(dataset) - total_scorable
    if nb_traps > 0:
        trap_acc = (trap_success_count / nb_traps) * 100
        print(f"🛡️  FILTRAGE DES PIÈGES    : {trap_acc:.2f}%")
    
    print("-" * 40)
    print("Détails par question :")
    for res in results:
        icon = "✅" if res["success"] else "❌"
        print(f"{icon} [Q{res['id']}] {res['info']}")
        
    # Sauvegarde CSV
    out_path = BASE_DIR / "data" / OUTPUT_FILENAME
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nRapport détaillé sauvegardé dans {out_path}")

if __name__ == "__main__":
    main()