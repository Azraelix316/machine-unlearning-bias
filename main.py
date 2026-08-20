import gc
import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Unbuffered logging setup for tmux
sys.stdout = open("unlearn_pipeline.log", "w", buffering=1)
sys.stderr = sys.stdout

HF_TOKEN = os.getenv("HF_TOKEN")

# ------------------------------------------------------------------------------
# STEP A: BIAS DETECTION (RoBERTa / DeBERTa Classifier)
# ------------------------------------------------------------------------------
def score_bias_with_roberta(
    texts, model_name="roberta-base", bias_threshold=0.6
):
    print(
        f"\n[STEP A] Scoring {len(texts)} samples with {model_name}...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    ).cuda()

    forget_set, retain_set = [], []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            bias_score = torch.softmax(logits, dim=-1)[0][1].item()

        if bias_score >= bias_threshold:
            forget_set.append(text)
        else:
            retain_set.append(text)

    # Free classifier from GPU memory before loading 32B/70B model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(
        f"[STEP A Complete] Mined {len(forget_set)} Biased (Forget) | {len(retain_set)} Clean (Retain) samples.",
        flush=True,
    )
    return forget_set, retain_set


# ------------------------------------------------------------------------------
# STEP B: DATA MINING FROM C4
# ------------------------------------------------------------------------------
def extract_c4_bias_data(sample_limit=10000):
    print("\n[STEP B] Streaming C4 dataset...", flush=True)
    dataset = load_dataset("c4", "en", streaming=True, split="train")

    raw_texts = []
    for i, item in enumerate(dataset):
        if i >= sample_limit:
            break
        raw_texts.append(item["text"][:512])

    return score_bias_with_roberta(raw_texts)


# ------------------------------------------------------------------------------
# STEP C: TARGET LLM UNLEARNING
# ------------------------------------------------------------------------------
def execute_unlearning(
    model_repo, forget_set, retain_set, quant_type="8bit"
):
    print(
        f"\n[STEP C] Loading target model {model_repo} ({quant_type})...",
        flush=True,
    )

    q_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if quant_type == "4bit"
        else BitsAndBytesConfig(load_in_8bit=True)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_repo, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_repo, quantization_config=q_config, device_map="auto", token=HF_TOKEN
    )

    # Gradient Ascent Unlearning Step on Biased Data
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    print("\nStarting Gradient Ascent on Forget Set...", flush=True)
    for epoch, text in enumerate(forget_set):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])

        # Invert loss to unlearn biased tokens
        unlearn_loss = -outputs.loss
        unlearn_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            f"Step {epoch+1}/{len(forget_set)} | Ascent Loss: {unlearn_loss.item():.4f}",
            flush=True,
        )

    print("\n[Complete] Unlearning pass finished successfully.", flush=True)


if __name__ == "__main__":
    forget, retain = extract_c4_bias_data(sample_limit=50)
    if forget:
        execute_unlearning(
            "Qwen/Qwen2.5-32B-Instruct",
            forget,
            retain,
            quant_type="8bit",
        )
    else:
        print("No biased samples found above threshold. Adjust threshold.")