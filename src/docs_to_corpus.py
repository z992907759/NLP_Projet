# IMPORTS
from pathlib import Path
import sys
import re
import json
import pandas as pd
import csv

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer




from pathlib import Path

sys.path.append("..") 

# CONFIGURATION DES CHEMINS
BASE_DIR = Path("..") 

# Dossier d'entrée
RAW_DOC_DIR = BASE_DIR / "data" / "raw_docs"

# Dossier de sortie
PROC_DIR = BASE_DIR / "data" / "processed"

# On crée le dossier s'il n'existe pas
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Fichier final
OUT_PATH = PROC_DIR / "docs_corpus.csv"



# FONCTIONS D'EXTRACTION (READERS)

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
            # Si aucune clé connue, on dump l'objet en JSON string
            texts.append(json.dumps(obj, ensure_ascii=False))
        else:
            texts.append(json.dumps(obj, ensure_ascii=False))

    try:
        # Cas 1 : JSON Lines (.jsonl) : Un objet JSON par ligne
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
        # Cas 2 : JSON Standard (.json) : Liste ou Objet unique
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






# FONCTIONS DE NETTOYAGE ET DÉCOUPAGE (PRE-PROCESSING)

def basic_clean(text: str) -> list[str]:
    # Normalisation des retours à la ligne
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Séparation en paragraphes en utilisant les lignes vides comme séparateurs
    raw_paragraphs = re.split(r"\n\s*\n", text)

    paragraphs: list[str] = []
    for para in raw_paragraphs:
        # Fusionner les espaces vides en 1
        para = re.sub(r"\s+", " ", para).strip()
        
        # Filtre de qualité : On ignore les headers, numéros de page, ou bruit < 20 caractères
        if len(para) < 20:
            continue
        paragraphs.append(para)

    return paragraphs

import re
from typing import List

def optimized_chunk_text(paragraphs: list[str], max_words: int = 300, overlap_words: int = 50):

    # 1. Text reconstruction
    full_text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    
    # 2. Sentence splitting (preserving punctuation)
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(full_text)
    
    chunks: list[str] = []
    current_chunk_sentences: list[str] = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
            
        sentence_word_count = len(sentence.split())

        # CASE A: GIANT SENTENCE
        # If the sentence alone exceeds the limit, we cut it
        if sentence_word_count > max_words:
            # 1. Save what we have already accumulated in the current buffer
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_word_count = 0
            
            # 2. Cut the giant sentence into pieces of size max_words
            words = sentence.split()
            for j in range(0, len(words), max_words - overlap_words):
                # Create a sub-chunk
                sub_chunk_words = words[j : j + max_words]
                chunks.append(" ".join(sub_chunk_words))
            
            # 3. Move to the next sentence
            i += 1
            continue

        # CASE B: STANDARD ADDITION
        # If it fits in the current chunk
        if current_word_count + sentence_word_count <= max_words:
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count
            i += 1
        
        # CASE C: CHUNK IS FULL (Overlap Handling)
        else:
            # 1. Validate (save) the current chunk
            chunks.append(" ".join(current_chunk_sentences))
            
            # 2. Calculate overlap (keep the last few sentences for context)
            overlap_buffer = []
            overlap_count = 0
            
            for prev_sent in reversed(current_chunk_sentences):
                prev_len = len(prev_sent.split())
                if overlap_count + prev_len <= overlap_words:
                    overlap_buffer.insert(0, prev_sent)
                    overlap_count += prev_len
                else:
                    break
            
            # 3. INFINITE LOOP SAFETY
            # If Overlap + New Sentence > Max, we cannot do a clean overlap
            # We must start fresh to avoid an infinite loop
            if overlap_count + sentence_word_count > max_words:
                current_chunk_sentences = [] # Abandon overlap to prevent blocking
                current_word_count = 0
            else:
                current_chunk_sentences = overlap_buffer[:]
                current_word_count = overlap_count

    # End of loop: save the remainder
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks








import csv
import re
import fitz  # PyMuPDF
from pathlib import Path

# NEW PDF READING FUNCTION
def read_pdf(path: Path) -> str:
    text_parts = []
    try:
        # 'fitz' (PyMuPDF) is much faster than PyPDF2 and does not hang on complex files
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {path.name}: {e}")
        return ""
    return "\n\n".join(text_parts)


# --- OPTIMIZED PIPELINE (Memory & Speed) ---
def main():
    # 1. Initialize output file with headers
    if not PROC_DIR.exists():
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        
    csv_headers = ["doc_id", "chunk_id", "text", "source"]
    
    # Overwrite if file exists to start fresh
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    doc_id = 1
    total_chunks = 0
    
    # Sort files to ensure constant and predictable processing order
    files = sorted(list(RAW_DOC_DIR.iterdir()))
    print(f"[INFO] Scanning {len(files)} files in {RAW_DOC_DIR}...")

    # 2. Process file by file
    for path in files:
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        
        # Reader selection based on file extension
        if suffix == ".txt":
            raw_text = read_txt(path)
        elif suffix == ".pdf":
            raw_text = read_pdf(path) # Now uses PyMuPDF
        elif suffix in {".json", ".jsonl"}:
            raw_text = read_json_as_text(path)
        else:
            continue
            
        # Basic Cleaning
        paragraphs = basic_clean(raw_text)
        if not paragraphs: 
            continue

        # Chunking (using the robust function)
        chunks = optimized_chunk_text(paragraphs, max_words=300)
        if not chunks: 
            continue

        # 3. Immediate writing to disk (Memory Safe)
        rows_to_write = []
        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            rows_to_write.append([doc_id, chunk_idx, chunk_text, path.name])
        
        # Open, write, close. RAM remains empty.
        with open(OUT_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)
        
        # Simple visual feedback
        print(f"[INFO] Processed {path.name}: {len(chunks)} chunks saved.")
        
        total_chunks += len(chunks)
        doc_id += 1

    print(f"\n[OK] Pipeline finished. {total_chunks} chunks written to {OUT_PATH}")

# Run the pipeline
main()