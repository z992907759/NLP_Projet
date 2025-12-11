from pathlib import Path
import textwrap

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

# ====== Load LLM ======

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading LLM model: {MODEL_NAME}")

# Select appropriate dtype based on available device
if torch.cuda.is_available() or torch.backends.mps.is_available():
    llm_dtype = torch.bfloat16
else:
    llm_dtype = torch.float32

# Load tokenizer and model (only loaded once at startup)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=llm_dtype,
    device_map="auto",  # Automatically select CPU / GPU / MPS
)

if hasattr(model, "generation_config"):
    try:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
    except Exception:
        pass


def call_llm(prompt: str) -> str:
    # 构造对话消息：system + user
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and precise research assistant. "
                "You must follow the instructions in the user's message and answer the question "
                "using ONLY the information from the provided CONTEXTS when possible. "
                "If the contexts do not contain the answer, or the information is incomplete or ambiguous, "
                "you must say that you are not sure instead of guessing."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # 使用聊天模板编码
    if hasattr(tokenizer, "apply_chat_template"):
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        # 否则手动拼一个简单的 chat 文本
        chat_text = (
            "System: You are a helpful and precise research assistant.\n"
            "User:\n"
            f"{prompt}\n\n"
            "Assistant:"
        )
        model_inputs = tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        )

    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs}

    # 避免警告
    if "attention_mask" not in model_inputs and "input_ids" in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])

    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的部分
    generated_ids = output_ids[0][model_inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if not answer:
        full_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        answer = full_text

    if not answer:
        answer = "The model did not generate a valid answer."

    # 打印一行简短调试信息
    print("\n[DEBUG] LLM answer snippet:\n", answer[:300].replace("\n", " "))

    return answer


# Baseline LLM answer without RAG context
def call_llm_baseline(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and knowledgeable assistant. "
                "Answer the user's question as clearly and accurately as possible."
            ),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        chat_text = (
            "System: You are a helpful and knowledgeable assistant.\n"
            "User:\n"
            f"{query}\n\n"
            "Assistant:"
        )
        model_inputs = tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        )

    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs}

    if "attention_mask" not in model_inputs and "input_ids" in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])

    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][model_inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if not answer:
        full_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        answer = full_text

    if not answer:
        answer = "The model did not generate a valid answer."

    print("\n[DEBUG] Baseline LLM answer snippet:\n", answer[:300].replace("\n", " "))

    return answer


BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index"


def load_resources():
    corpus_path = PROC_DIR / "docs_corpus.csv"
    df = pd.read_csv(corpus_path)

    faiss_path = INDEX_DIR / "corpus.index"
    index = faiss.read_index(str(faiss_path))

    model_name_path = INDEX_DIR / "embedding_model.txt"
    model_name = model_name_path.read_text(encoding="utf-8").strip()
    print(f"Loaded embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name)

    return df, index, embed_model


def retrieve(query: str, df, index, embed_model, top_k: int = 5):
    query_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores, indices = index.search(query_emb, top_k)

    contexts = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        text = str(row["text"])
        doc_id = row.get("doc_id", idx)
        contexts.append(
            {
                "doc_id": doc_id,
                "score": float(score),
                "text": text,
            }
        )
    return contexts


def build_prompt(query: str, contexts):
    context_blocks = []
    for i, ctx in enumerate(contexts, start=1):
        block = textwrap.dedent(
            f"""\
            [Context {i} | score={ctx["score"]:.4f}]
            {ctx["text"]}
            """
        ).strip()
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    prompt = textwrap.dedent(
        f"""
            CONTEXTS:
            {context_str}

            QUESTION:
            {query}

            ANSWER INSTRUCTIONS:
            - Use ONLY the information from the contexts above. Do not use outside knowledge.
            - First, give a concise direct answer to the question (1–3 sentences).
            - Then, if useful, add a short explanation or a brief numbered list of key points.
            - If the contexts do not clearly contain the answer, say explicitly that you are not sure and explain why.
            """
    ).strip()

    return prompt


def rag_answer(query: str, df, index, embed_model, top_k: int = 5) -> str:
    contexts = retrieve(query, df, index, embed_model, top_k=top_k)
    print("\n[DEBUG] Retrieved contexts:")
    for i, ctx in enumerate(contexts, start=1):
        preview = ctx["text"][:150].replace("\n", " ")
        print(f"  ({i}) doc_id={ctx['doc_id']} score={ctx['score']:.4f}")
        print(f"      {preview}...")

    # If no contexts at all, fall back directly to baseline
    if not contexts:
        baseline_answer = call_llm_baseline(query)
        return ("****************************[RAG NOTICE] No relevant information was found in the vector database. "
                "Falling back to the base model's own knowledge.****************************\n\n") + baseline_answer

    # Compute maximum similarity score
    max_score = max(ctx["score"] for ctx in contexts)
    if max_score < 0.5:
        # Similarity too low: treat contexts as irrelevant and fall back
        baseline_answer = call_llm_baseline(query)
        return ("****************************[RAG NOTICE] No relevant information was found in the vector database."
                " Falling back to the base model's own knowledge.****************************\n\n") + baseline_answer

    # Otherwise, use RAG as normal
    prompt = build_prompt(query, contexts)
    answer = call_llm(prompt)
    return answer


def retrieve_multi_query(query: str, df, index, embed_model, top_k: int = 5):
    """
    Recherche avancée avec reformulation
    """
    # Prompt pour générer des variations (Reformulation)
    prompt_reformulation = (
        f"You are an AI language model assistant. Your task is to generate 3 different versions "
        f"of the given user question to retrieve relevant documents from a vector database. "
        f"Provide these alternative questions separated by newlines. "
        f"Original question: {query}"
    )
    
    # On utilise le modèle pour imaginer d'autres façons de poser la question
    print(f"\n[MULTI-QUERY] Generating variations for: '{query}'...")
    variations_text = call_llm_baseline(prompt_reformulation)
    
    # Parsing : on nettoie pour avoir une liste de questions
    queries = [query] # On garde l'originale
    for line in variations_text.split('\n'):
        line = line.strip()
        if line and "?" in line: 
            queries.append(line)
            
    # On limite pour ne pas être trop lent
    queries = queries[:4] 
    print(f"[MULTI-QUERY] Questions utilisées : {queries}")

    # Boucle de recherche et Dédoublonnage
    unique_docs = {} 
    
    for q in queries:
        # On cherche k=3 pour chaque variation
        results = retrieve(q, df, index, embed_model, top_k=3)
        
        for doc in results:
            # On utilise le texte comme clé unique pour dédoublonner
            key = doc['text']
            if key not in unique_docs:
                unique_docs[key] = doc
                
    # On retourne la liste dédoublonnée
    final_contexts = list(unique_docs.values())
    return final_contexts[:top_k*2]

def rag_answer_multiquery(query: str, df, index, embed_model, top_k: int = 5) -> str:
    """
     Multi-Query Retrieval
    """
    # Récupération avancée (Reformulation + Dédoublonnage)
    contexts = retrieve_multi_query(query, df, index, embed_model, top_k=top_k)

    # Affichage des sources trouvées
    print("\n[DEBUG] Contextes récupérés via Multi-Query :")
    for i, ctx in enumerate(contexts, start=1):
        # On coupe le texte pour que ça reste lisible dans la console
        preview = ctx["text"][:150].replace("\n", " ")
        # On affiche le score (qui vient de la requête spécifique qui a trouvé ce doc)
        print(f"  ({i}) doc_id={ctx.get('doc_id', '?')} score={ctx.get('score', 0.0):.4f}")
        print(f"      {preview}...")

    # Vérification : Si vide ou scores trop bas
    if not contexts:
        print("[WARN] Aucun document trouvé après reformulation.")
        return call_llm_baseline(query)

    # On vérifie le meilleur score parmi tous les docs trouvés
    max_score = max(ctx["score"] for ctx in contexts)
    
    if max_score < 0.45: 
        print(f"[WARN] Score trop faible ({max_score:.4f}) malgré Multi-Query -> Baseline.")
        baseline = call_llm_baseline(query)
        return f"[RAG NOTICE] Information introuvable (Score max: {max_score:.2f}).\n\n{baseline}"

    # Génération de la réponse
    prompt = build_prompt(query, contexts)
    answer = call_llm(prompt)
    return answer


def main():
    df, index, embed_model = load_resources()

    print("============== RAG QA demo ==============")
    print("Type q / quit to exit.")

    while True:
        query = input("\nPlease enter your question: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            print("Bye ~")
            break

        answer = rag_answer_multiquery(query, df, index, embed_model, top_k=5)
        print("\n=================== Model Answer ===================")
        print(answer)
        print("="*50)


if __name__ == "__main__":
    main()