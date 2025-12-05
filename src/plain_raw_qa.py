from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ====== Load LLM model ======
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading LLM model: {MODEL_NAME}")

# Select appropriate dtype based on available device
if torch.cuda.is_available() or torch.backends.mps.is_available():
    llm_dtype = torch.bfloat16
else:
    llm_dtype = torch.float32

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=llm_dtype,
    device_map="auto",  # Automatically choose CPU / GPU / MPS
)


def call_llm_raw(text: str) -> str:
    """
    Pure raw version: no prompt added, directly send user input to the model.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if not answer:
        answer = "The model did not generate a valid answer."

    return answer


def main():
    print("=== Pure LLM raw QA demo (no RAG, no prompt) ===")
    print("Type q / quit to exit.")

    while True:
        query = input("\nPlease enter your question: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            print("Bye ~")
            break

        answer = call_llm_raw(query)

        print("\n===== Model answer (raw) =====")
        print(answer)
        print("================================")


if __name__ == "__main__":
    main()