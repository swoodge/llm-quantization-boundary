# eval_perplexity.py
import json
import math
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def compute_perplexity(model, tokenizer, texts, stride=512, max_length=1024, device="cuda"):
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.shape[1]

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end
            input_chunk = input_ids[:, begin:end]
            target_ids = input_chunk.clone()
            target_ids[:, :-target_len] = -100

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_ids)
                nll = outputs.loss * target_len

            total_nll += nll.item()
            total_tokens += target_len
            prev_end = end

            if end == seq_len:
                break

    ppl = math.exp(total_nll / total_tokens)
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id",   required=True, help="Короткое имя для логов")
    parser.add_argument("--output",     default="results/perplexity.jsonl")
    parser.add_argument("--n_samples",  type=int, default=50)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    # Датасет: WikiText-2 — стандарт для perplexity
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.strip()) > 200][:args.n_samples]

    print(f"Loading {args.model_id} from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    ppl = compute_perplexity(model, tokenizer, texts, device=args.device)
    print(f"[{args.model_id}] Perplexity: {ppl:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a") as f:
        f.write(json.dumps({"model_id": args.model_id, "perplexity": ppl}) + "\n")

if __name__ == "__main__":
    main()
