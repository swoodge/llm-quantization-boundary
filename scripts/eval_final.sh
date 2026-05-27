#!/bin/bash
#SBATCH --job-name=eval_final
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/eval_final_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/eval_final_%j.err

module load gnu12/12.1
module load cmake/3.31.8
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

echo "=== PPL: qwen2.5-14b-Q3_K_M ===" | tee -a ./results/ppl_all.txt
./llama.cpp/build_cpu/bin/llama-perplexity \
    -m ./models/qwen2.5-14b-Q3_K_M.gguf \
    -f ./data/wikitext2_test.txt \
    --ctx-size 512 -t 16 \
    2>&1 | tee -a ./results/ppl_all.txt | grep "Final"

echo "=== fp16 baselines via HF transformers ===" | tee -a ./results/ppl_fp16.txt

python3 - << 'PYEOF'
import math, json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def compute_ppl(model_path, n_samples=30, stride=512, max_len=1024):
    print(f"Loading {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:n_samples]

    total_nll, total_tok = 0.0, 0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=False).input_ids
        seq_len = ids.shape[1]
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_len, seq_len)
            target_len = end - prev_end
            chunk = ids[:, begin:end].cuda()
            tgt = chunk.clone()
            tgt[:, :-target_len] = -100
            with torch.no_grad():
                loss = model(chunk, labels=tgt).loss
            total_nll += loss.item() * target_len
            total_tok += target_len
            prev_end = end
            if end == seq_len:
                break
    return math.exp(total_nll / total_tok)

models = {
    "qwen2.5-7b-fp16":  "./hf_models/Qwen2.5-7B",
    "qwen2.5-14b-fp16": "./hf_models/Qwen2.5-14B",
}

Path("results").mkdir(exist_ok=True)
for name, path in models.items():
    if not Path(path).exists():
        print(f"[skip] {path} not found")
        continue
    ppl = compute_ppl(path)
    result = {"model_id": name, "perplexity": ppl}
    print(f"{name}: PPL = {ppl:.4f}")
    with open("results/ppl_fp16.txt", "a") as f:
        f.write(json.dumps(result) + "\n")
PYEOF

echo "=== DONE ===" | tee -a ./results/ppl_all.txt
