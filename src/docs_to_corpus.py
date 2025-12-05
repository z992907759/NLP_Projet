from pathlib import Path
import re
import json
import pandas as pd

# Directory containing raw documents (plain text)
RAW_DOC_DIR = Path("../data/raw_docs")
PROC_DIR = Path("../data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = PROC_DIR / "docs_corpus.csv"


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
            # Prefer common text fields
            for key in ("text", "content", "body"):
                if key in obj and isinstance(obj[key], str):
                    texts.append(obj[key])
                    return
            # As a fallback, dump the whole object as JSON
            texts.append(json.dumps(obj, ensure_ascii=False))
        else:
            texts.append(json.dumps(obj, ensure_ascii=False))

    try:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    _extract_from_obj(obj)
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


def basic_clean(text: str) -> list[str]:
    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into raw paragraphs using empty lines as separators
    raw_paragraphs = re.split(r"\n\s*\n", text)

    paragraphs: list[str] = []
    for para in raw_paragraphs:
        # Collapse internal whitespace
        para = re.sub(r"\s+", " ", para).strip()
        # Drop very short paragraphs (headers, noise, etc.)
        if len(para) < 20:
            continue
        paragraphs.append(para)

    return paragraphs


def chunk_text(paragraphs: list[str], max_words: int = 300):
    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        words = para.split()

        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words]).strip()
                if chunk:
                    chunks.append(chunk)
            continue

        if len(current_words) + len(words) <= max_words:
            current_words.extend(words)
        else:
            chunk = " ".join(current_words).strip()
            if chunk:
                chunks.append(chunk)
            current_words = words[:]

    if current_words:
        chunk = " ".join(current_words).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def main():
    rows = []
    doc_id = 1

    if not RAW_DOC_DIR.exists():
        print(f"[ERROR] RAW_DOC_DIR does not exist: {RAW_DOC_DIR}")
        return

    for path in RAW_DOC_DIR.iterdir():
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".txt":
            raw_text = read_txt(path)
        elif suffix == ".pdf":
            raw_text = read_pdf(path)
        elif suffix in {".json", ".jsonl"}:
            raw_text = read_json_as_text(path)
        else:
            print(f"[WARN] Skipping unsupported file type: {path}")
            continue
        paragraphs = basic_clean(raw_text)
        if not paragraphs:
            print(f"[WARN] Empty document after cleaning: {path}")
            continue

        chunks = chunk_text(paragraphs, max_words=300)
        if not chunks:
            print(f"[WARN] No chunks produced for: {path}")
            continue

        for chunk_id, chunk in enumerate(chunks, start=1):
            rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source": path.name,
                }
            )

        print(f"[INFO] Processed {path.name}: {len(chunks)} chunks")
        doc_id += 1

    if not rows:
        print("[ERROR] No chunks generated. Check documents in data/raw_docs.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(df)} chunks from {doc_id-1} documents to {OUT_PATH}")


if __name__ == "__main__":
    main()