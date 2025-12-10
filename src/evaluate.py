import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# On importe tes fonctions depuis main.py
# Assure-toi que main.py est bien dans le même dossier src/
try:
    from main import load_resources, retrieve
except ImportError:
    # Fallback si lancé depuis la racine
    import sys
    sys.path.append("src")
    from main import load_resources, retrieve

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "golden_dataset.json"

def calculate_metrics(retrieved_docs, expected_ids):
    """
    Vérifie si l'un des documents attendus est présent dans les résultats récupérés.
    C'est ce qu'on appelle le "Recall@k".
    """
    if not expected_ids:
        return False, 0.0 # Cas des questions pièges (traitées à part)
    
    found_ids = [d['doc_id'] for d in retrieved_docs]
    
    # Est-ce qu'on a trouvé au moins UN bon document ?
    success = any(e_id in found_ids for e_id in expected_ids)
    
    # Calcul du score moyen des bons documents trouvés
    relevant_scores = [d['score'] for d in retrieved_docs if d['doc_id'] in expected_ids]
    avg_score = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
    
    return success, avg_score

def main():
    print("=== Démarrage de l'évaluation automatique ===")
    
    # 1. Chargement du Golden Dataset
    if not DATASET_PATH.exists():
        print(f"[ERREUR] Le fichier {DATASET_PATH} n'existe pas.")
        return
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] {len(dataset)} questions chargées depuis le Golden Dataset.")

    # 2. Chargement du moteur RAG
    print("[INFO] Chargement des ressources RAG (Index, Modèle...)...")
    df, index, embed_model = load_resources()
    
    # 3. Boucle d'évaluation
    results = []
    correct_retrieval_count = 0
    trap_success_count = 0
    
    total_scorable = 0 # On ne compte pas les pièges dans le score de retrieval pur
    
    print("\n--- Lancement des tests ---")
    for item in tqdm(dataset):
        query = item["question"]
        q_type = item["type"]
        expected_ids = item.get("expected_doc_ids", [])
        
        # Lancer la recherche (Top 5)
        retrieved_contexts = retrieve(query, df, index, embed_model, top_k=5)
        
        # Analyse des résultats
        if q_type == "trap":
            # Pour un piège, succès = scores très bas (tous < 0.5 par exemple)
            max_score = max([c['score'] for c in retrieved_contexts]) if retrieved_contexts else 0
            is_success = max_score < 0.5
            if is_success:
                trap_success_count += 1
            
            results.append({
                "id": item["id"],
                "type": "trap",
                "success": is_success,
                "info": f"Max Score: {max_score:.4f} (Seuil: 0.5)"
            })
            
        else:
            # Pour une question normale
            total_scorable += 1
            is_success, avg_conf = calculate_metrics(retrieved_contexts, expected_ids)
            
            if is_success:
                correct_retrieval_count += 1
            
            # On note quel ID a été trouvé pour le débug
            found_ids = [c['doc_id'] for c in retrieved_contexts]
            results.append({
                "id": item["id"],
                "type": q_type,
                "success": is_success,
                "info": f"Attendu: {expected_ids} | Trouvé: {found_ids}"
            })

    # 4. Génération du Rapport
    accuracy = (correct_retrieval_count / total_scorable * 100) if total_scorable > 0 else 0
    
    print("\n" + "="*40)
    print("       RAPPORT D'ÉVALUATION RAG       ")
    print("="*40)
    print(f"Questions valides testées : {total_scorable}")
    print(f"Questions pièges testées  : {len(dataset) - total_scorable}")
    print("-" * 40)
    print(f"✅ PRÉCISION DU RETRIEVAL : {accuracy:.2f}%")
    print("-" * 40)
    
    if len(dataset) - total_scorable > 0:
        trap_acc = (trap_success_count / (len(dataset) - total_scorable)) * 100
        print(f"🛡️  FILTRAGE DES PIÈGES    : {trap_acc:.2f}%")
        print("(Capacité à détecter le hors-sujet via le seuil)")
    
    print("-" * 40)
    print("Détails par question :")
    for res in results:
        icon = "✅" if res["success"] else "❌"
        print(f"{icon} [Q{res['id']} - {res['type']}] {res['info']}")
        
    # Sauvegarde CSV
    pd.DataFrame(results).to_csv(BASE_DIR / "data" / "evaluation_results.csv", index=False)
    print(f"\nRapport détaillé sauvegardé dans data/evaluation_results.csv")

if __name__ == "__main__":
    main()