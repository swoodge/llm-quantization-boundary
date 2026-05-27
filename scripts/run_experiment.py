#!/usr/bin/env python3
# run_experiment.py
"""
Запускает полный цикл: квантизация → оценка → сохранение результатов
для всех моделей и форматов.
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class ModelConfig:
    family: str
    size_b: float          # миллиарды параметров
    hf_path: str           # путь к fp16 модели
    short_name: str        # для имён файлов

MODELS = [
    ModelConfig("qwen2.5", 0.5,  "./hf_models/Qwen2.5-0.5B", "qwen2.5-0.5b"),
    ModelConfig("qwen2.5", 1.5,  "./hf_models/Qwen2.5-1.5B", "qwen2.5-1.5b"),
    ModelConfig("qwen2.5", 3.0,  "./hf_models/Qwen2.5-3B",   "qwen2.5-3b"),
    ModelConfig("qwen2.5", 7.0,  "./hf_models/Qwen2.5-7B",   "qwen2.5-7b"),
    ModelConfig("qwen2.5", 14.0, "./hf_models/Qwen2.5-14B",  "qwen2.5-14b"),
    ModelConfig("qwen2.5", 32.0, "./hf_models/Qwen2.5-32B",  "qwen2.5-32b"),
    ModelConfig("qwen2.5", 72.0, "./hf_models/Qwen2.5-72B",  "qwen2.5-72b"),
]

QUANT_TYPES = ["fp16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

RESULTS_FILE = Path("./results/experiment_results.jsonl")
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

def quantize_model(model: ModelConfig, quant_type: str) -> Path:
    """Возвращает путь к готовому GGUF файлу."""
    out_path = Path(f"./models/{model.short_name}-{quant_type}.gguf")
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return out_path

    print(f"  [quantize] {model.short_name} → {quant_type}")
    subprocess.run(
        ["bash", "quantize_all.sh", model.hf_path, model.short_name],
        check=True
    )
    return out_path

def eval_perplexity(model_path: str, model_id: str) -> float:
    """Запускает eval_perplexity.py и возвращает число."""
    result = subprocess.run(
        ["python3", "eval_perplexity.py",
         "--model_path", model_path,
         "--model_id", model_id,
         "--n_samples", "50"],
        capture_output=True, text=True, check=True
    )
    # Ищем строку вида: "[qwen2.5-7b-Q4_K_M] Perplexity: 8.1234"
    for line in result.stdout.splitlines():
        if "Perplexity:" in line:
            return float(line.split("Perplexity:")[-1].strip())
    raise RuntimeError(f"No perplexity in output:\n{result.stdout}")

def compute_degradation(baseline_ppl: float, quant_ppl: float) -> float:
    """Деградация в % (рост perplexity относительно baseline)."""
    return (quant_ppl - baseline_ppl) / baseline_ppl * 100

def main():
    all_results = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model.short_name}  ({model.size_b}B params)")
        print(f"{'='*60}")

        baseline_ppl = None
        model_results = {"model": asdict(model), "quant_results": []}

        for quant in QUANT_TYPES:
            model_id = f"{model.short_name}-{quant}"

            # Для fp16 используем HF путь напрямую
            if quant == "fp16":
                model_path = model.hf_path
            else:
                gguf_path = Path(f"./models/{model.short_name}-{quant}.gguf")
                if not gguf_path.exists():
                    print(f"  [warn] {gguf_path} not found, skipping")
                    continue
                model_path = str(gguf_path)

            try:
                ppl = eval_perplexity(model_path, model_id)
            except Exception as e:
                print(f"  [error] {model_id}: {e}")
                continue

            if quant == "fp16":
                baseline_ppl = ppl
                degradation = 0.0
            else:
                degradation = compute_degradation(baseline_ppl, ppl)

            result = {
                "model_id": model_id,
                "quant_type": quant,
                "perplexity": ppl,
                "degradation_pct": degradation,
            }
            model_results["quant_results"].append(result)
            print(f"  {quant:12s}  PPL={ppl:.4f}  Δ={degradation:+.2f}%")

        all_results.append(model_results)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(model_results) + "\n")

    print(f"\n\nAll results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
