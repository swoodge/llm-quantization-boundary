#!/bin/bash
#SBATCH --job-name=fp16_baseline
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/fp16_baseline_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/fp16_baseline_%j.err

module load gnu12/12.1
module load Python/PyTorch_GPU_v2.4

cd ~/quant_experiment

python3 - << 'PYEOF'
import math, json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def compute_ppl(model_path, n_samples=20):
    print(f"Loading {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True)
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:n_samples]

    total_nll, total_tok = 0.0, 0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=False).input_ids
        seq_len = ids.shape[1]
        stride, max_len = 512, 1024
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_len, seq_len)
            target_len = end - prev_end
            chunk = ids[:, begin:end]
            tgt = chunk.clone()
            tgt[:, :-target_len] = -100
            with torch.no_grad():
                loss = model(chunk, labels=tgt).loss
            total_nll += loss.item() * target_len
            total_tok += target_len
            prev_end = end
            if end == seq_len:
                break
    ppl = math.exp(total_nll / total_tok)
    return ppl

models = {
    "qwen2.5-0.5b-fp16": "./hf_models/Qwen2.5-0.5B",
    "qwen2.5-1.5b-fp16": "./hf_models/Qwen2.5-1.5B",
    "qwen2.5-3b-fp16":   "./hf_models/Qwen2.5-3B",
    "qwen2.5-14b-fp16":  "./hf_models/Qwen2.5-14B",
}

Path("results").mkdir(exist_ok=True)
for name, path in models.items():
    if not Path(path).exists():
        print(f"[skip] {path}")
        continue
    try:
        ppl = compute_ppl(path)
        print(f"{name}: PPL = {ppl:.4f}")
        with open("results/ppl_fp16.txt", "a") as f:
            f.write(json.dumps({"model_id": name, "perplexity": round(ppl, 4)}) + "\n")
    except Exception as e:
        print(f"ERROR {name}: {e}")
PYEOF

echo "=== DONE ==="
