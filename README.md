# NLP Project – RAG-based Agricultural QA

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for agricultural question answering using a local LLM (**Llama 3.2 1B Instruct**) and a FAISS vector index built from an English agriculture QA corpus.

---

## Project Structure

```text
data/
  raw/            # Original datasets
  processed/      # Cleaned corpus + evaluation set
  index/          # FAISS index + embeddings

src/
  clean_data.py       # merge & clean raw datasets
  prepare_rag_data.py # build RAG corpus + eval set
  build_index.py      # encode corpus + build FAISS index
  search_index.py     # simple retrieval demo
  plain_raw_qa.py     # baseline: LLM only (no RAG)
  rag_qa.py           # main RAG QA pipeline
```

## Environment
```
conda create -n NLP python=3.10
conda activate NLP

pip install torch transformers sentence-transformers faiss-cpu pandas numpy
```
Model required (Hugging Face):
```
meta-llama/Llama-3.2-1B-Instruct
```

## Pipeline
###  1.	Clean and merge raw data
```
python src/clean_data.py
```

###  2.	Prepare RAG corpus and evaluation set
```
python src/prepare_rag_data.py
```

###  3.	Build FAISS index
```
python src/build_index.py
```

###  4.	(Optional)Test retrieval only
```
python src/search_index.py
```

###  5.	RAG QA demo (LLM + retrieval)
```
python src/main.py
```

###  6.	Baseline demo (LLM without RAG)
```
python src/plain_raw_qa.py
```

##  How RAG Is Used
When a user asks a question:
	1.	The system encodes the query using SentenceTransformer
	2.	Retrieves the top-k relevant documents from the FAISS index
	3.	Inserts them into a structured prompt:
```
CONTEXTS:
[Context i | score=...]
...

QUESTION:
<user question>

ANSWER INSTRUCTIONS:
- Use ONLY the provided contexts.
- Answer in the same language as the question.
- If contexts do not contain the answer:
  reply with one short sentence saying you are not sure.
```

##  Course Note
This project implements a complete NLP pipeline:
	- dataset cleaning
	- corpus construction
	- sentence embeddings
	- FAISS retrieval
	- structured prompting
	- LLM generation

The advanced topic chosen is Retrieval-Augmented Generation (RAG), and a baseline model is provided for comparison.
