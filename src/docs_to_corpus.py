from pathlib import Path
import re
import json
import pandas as pd

# CONFIGURATION DES CHEMINS (ETL SETUP)

# On remonte d'un niveau (../) pour atteindre la racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent 

# Dossier d'entrée (Documents bruts)
# Assure-toi que tes PDFs/TXT sont bien ici
RAW_DOC_DIR = BASE_DIR / "data" / "raw"

# Dossier de sortie (Données traitées)
PROC_DIR = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Fichier final (Corpus structuré)
OUT_PATH = PROC_DIR / "docs_corpus.csv"


# 1. FONCTIONS D'EXTRACTION (READERS)

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:

    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("[ERROR] PyPDF2 is not installed. Please run: pip install PyPDF2")
        return ""

    text_parts: list[str] = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            # extract_text() peut renvoyer None sur des pages vides/images
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {path}: {e}")
        return ""

    return "\n\n".join(text_parts)


def read_json_as_text(path: Path) -> str:
    texts: list[str] = []

    def _extract_from_obj(obj):
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            # Stratégie heuristique : on cherche les clés communes contenant du texte
            for key in ("text", "content", "body"):
                if key in obj and isinstance(obj[key], str):
                    texts.append(obj[key])
                    return
            # Fallback : si aucune clé connue, on dump l'objet en JSON string
            texts.append(json.dumps(obj, ensure_ascii=False))
        else:
            texts.append(json.dumps(obj, ensure_ascii=False))

    try:
        # Cas 1 : JSON Lines (.jsonl) - Un objet JSON par ligne
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    _extract_from_obj(obj)
        # Cas 2 : JSON Standard (.json) - Liste ou Objet unique
        else:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    _extract_from_obj(item)
            else:
                _extract_from_obj(data)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON {path}: {e}")
        return ""

    return "\n\n".join(texts)


# 2. FONCTIONS DE NETTOYAGE & DÉCOUPAGE (PRE-PROCESSING)

def basic_clean(text: str) -> list[str]:
    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into raw paragraphs using empty lines as separators
    raw_paragraphs = re.split(r"\n\s*\n", text)

    paragraphs: list[str] = []
    for para in raw_paragraphs:
        # Collapse internal whitespace (transforme les tabulations/espaces multiples en 1 espace)
        para = re.sub(r"\s+", " ", para).strip()
        
        # Filtre de qualité : On ignore les headers, numéros de page, ou bruit < 20 caractères
        if len(para) < 20:
            continue
        paragraphs.append(para)

    return paragraphs


def chunk_text(paragraphs: list[str], max_words: int = 300):
    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        words = para.split()

        # CAS 1 : Le paragraphe seul est DÉJÀ plus gros que la limite
        # On est obligé de le découper brutalement
        if len(words) > max_words:
            # Si on avait des mots en attente, on les sauvegarde d'abord
            if current_words:
                chunks.append(" ".join(current_words).strip())
                current_words = []
                
            # Découpage du gros paragraphe en sous-morceaux
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words]).strip()
                if chunk:
                    chunks.append(chunk)
            continue

        # CAS 2 : On peut ajouter ce paragraphe au chunk courant sans dépasser la limite
        if len(current_words) + len(words) <= max_words:
            current_words.extend(words)
        
        # CAS 3 : Le chunk courant est plein, on le sauvegarde et on en commence un nouveau
        else:
            chunk = " ".join(current_words).strip()
            if chunk:
                chunks.append(chunk)
            current_words = words[:] # On démarre le nouveau chunk avec le paragraphe actuel

    # Ne pas oublier le dernier morceau resté dans le buffer
    if current_words:
        chunk = " ".join(current_words).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# MAIN

def main():
    """
    Point d'entrée du script d'ingestion.
    Parcourt RAW_DOC_DIR -> Extrait -> Nettoie -> Chunk -> Sauvegarde CSV.
    """
    rows = []
    doc_id = 1

    # Vérification dossier
    if not RAW_DOC_DIR.exists():
        print(f"[ERROR] RAW_DOC_DIR does not exist: {RAW_DOC_DIR}")
        print("Please create the folder and put your PDF/TXT files inside.")
        return

    print(f"[INFO] Scanning files in {RAW_DOC_DIR}...")

    # Boucle sur les fichiers
    for path in RAW_DOC_DIR.iterdir():
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        # 1. Extraction selon le type
        if suffix == ".txt":
            raw_text = read_txt(path)
        elif suffix == ".pdf":
            raw_text = read_pdf(path)
        elif suffix in {".json", ".jsonl"}:
            raw_text = read_json_as_text(path)
        else:
            print(f"[WARN] Skipping unsupported file type: {path.name}")
            continue

        # 2. Nettoyage
        paragraphs = basic_clean(raw_text)
        if not paragraphs:
            print(f"[WARN] Empty document after cleaning: {path.name}")
            continue

        # 3. Chunking (Découpage)
        chunks = chunk_text(paragraphs, max_words=300)
        if not chunks:
            print(f"[WARN] No chunks produced for: {path.name}")
            continue

        # 4. Structuration des données
        for chunk_id, chunk in enumerate(chunks, start=1):
            rows.append(
                {
                    "doc_id": doc_id,       # ID unique du document
                    "chunk_id": chunk_id,   # ID du morceau dans le doc
                    "text": chunk,          # Contenu
                    "source": path.name,    # Méta-donnée source
                }
            )

        print(f"[INFO] Processed {path.name}: {len(chunks)} chunks")
        doc_id += 1

    # 5. Sauvegarde Finale
    if not rows:
        print("[ERROR] No chunks generated. Check documents in data/raw.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\n[OK] Pipeline Terminé.")
    print(f"     Saved {len(df)} chunks from {doc_id-1} documents to {OUT_PATH}")


if __name__ == "__main__":
    main()