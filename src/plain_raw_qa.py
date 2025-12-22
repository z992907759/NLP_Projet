from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# SCRIPT DE TEST : BASELINE (LLM SANS RAG)
# Ce script sert à démontrer les limites du modèle "nu".
# Il permet de montrer au jury que sans tes documents PDF,
# le modèle hallucine ou répond de manière trop générique.

# ====== CONFIGURATION ======
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

print(f"[INIT] Loading LLM model: {MODEL_NAME}")

# 1. Configuration Matérielle
# On vérifie si un GPU est disponible pour accélérer l'inférence.
if torch.cuda.is_available() or torch.backends.mps.is_available():
    llm_dtype = torch.bfloat16 # Mode rapide (16 bits)
    print("-> Mode: GPU Acceleration")
else:
    llm_dtype = torch.float32  # Mode standard (32 bits)
    print("-> Mode: CPU")

# 2. Chargement du Tokenizer et du Modèle
# Le Tokenizer transforme le texte en suite de nombres.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Le Modèle est chargé avec gestion automatique de la mémoire (device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=llm_dtype,
    device_map="auto", 
)


def call_llm_raw(text: str) -> str:
    # A. Tokenization
    # On convertit la question en tenseurs PyTorch ('pt')
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    
    # B. Transfert sur GPU (si dispo)
    # Il faut que les données soient au même endroit que le modèle
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # C. Génération
    with torch.no_grad(): # On désactive le calcul de gradients (économie mémoire)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,      # Longueur max de la réponse
            do_sample=False,         # False = Réponse déterministe (toujours la même)
            pad_token_id=tokenizer.eos_token_id,
        )

    # D. Décodage
    # On retire la question (input_ids) pour ne garder que la réponse générée
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True, # On enlève les balises techniques (<EOS>, <BOS>)
        clean_up_tokenization_spaces=True,
    ).strip()

    if not answer:
        answer = "The model did not generate a valid answer."

    return answer


def main():
    print("=== TEST : PURE LLM (NO RAG) ===")
    print("Ce script interroge le modèle sans accès à vos documents.")
    print("Utilisez-le pour mettre en évidence les hallucinations.")
    print("Tapez 'q' pour quitter.")

    while True:
        query = input("\nVotre question : ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            print("Bye ~")
            break

        # Appel direct (Raw)
        answer = call_llm_raw(query)

        print("\n===== Réponse du Modèle (Sans contexte) =====")
        print(answer)
        print("=============================================")


if __name__ == "__main__":
    main()