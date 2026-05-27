# Quantization Boundary Study: Qwen2.5 Family

Systematic study of post-training quantization (PTQ) quality degradation across model sizes.  
**Research question**: at what parameter count does quantization become harmful (>5% perplexity degradation)?

## Key Findings

| Format | Safe threshold | Notes |
|--------|---------------|-------|
| Q5_K_M | All sizes | <2.3% degradation everywhere |
| Q4_K_M | **≥7B** | 3B borderline (+5.0%), 7B safe (+1.9%) |
| Q3_K_M | **≥7B** | 3B catastrophic (+46.5%) |
| Q2_K   | **None** | Collapse at 3B (PPL=21454), critical at 1.5B (+137%) |

> ⚠️ **Critical finding**: Qwen2.5-3B with Q2_K quantization completely collapses (PPL = 21,454 vs baseline 7.95).

## Results

### Perplexity by model size and quantization format
(baseline = Q8_0, evaluated on WikiText-2)

| Model  | Q8_0  | Q5_K_M        | Q4_K_M        | Q3_K_M          | Q2_K              |
|--------|-------|---------------|---------------|-----------------|-------------------|
| 0.5B   | 13.12 | 13.36 (+1.8%) | 13.55 (+3.3%) | 14.17 (+8.0%)   | 15.76 (+20.1%)    |
| 1.5B   | 9.20  | 9.32 (+1.3%)  | 9.50 (+3.2%)  | 10.70 (+16.3%)  | 21.82 (+137.2%)   |
| 3B     | 7.95  | 8.13 (+2.3%)  | 8.35 (+5.0%)  | 11.65 (+46.5%)  | **COLLAPSE**      |
| 7B     | 6.74  | 6.82 (+1.2%)  | 6.87 (+1.9%)  | 7.25 (+7.6%)    | 9.44 (+40.1%)     |
| 14B    | 5.11  | —             | —             | —               | 7.43 (+45.4%)*    |

\* 14B Q2_K degradation relative to fp16 baseline (5.81)

## Setup

### Requirements
- llama.cpp (for GGUF conversion and quantization)
- Python 3.10+
- PyTorch, transformers, datasets

### Environment (HSE HPC cluster)
```bash
module load gnu12/12.1
module load CUDA/12.4
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH
```

### Reproduce

```bash
# 1. Download models
python3 -c "
from huggingface_hub import snapshot_download
for size in ['0.5B','1.5B','3B','7B','14B']:
    snapshot_download(f'Qwen/Qwen2.5-{size}', local_dir=f'./hf_models/Qwen2.5-{size}')
"

# 2. Build llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build_cpu -DGGML_CUDA=OFF && cmake --build build_cpu -j8

# 3. Convert and quantize
bash scripts/quantize_all.sh

# 4. Evaluate perplexity
sbatch scripts/eval_ppl.sh

# 5. Analyze results
python3 scripts/analyze_results.py
```

## Methodology

- **Models**: Qwen2.5 family (0.5B → 14B), same architecture across all sizes
- **Quantization**: GGUF format via llama.cpp (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q8_0)
- **Evaluation**: Perplexity on WikiText-2 test set, context length 512, stride 512
- **Baseline**: Q8_0 used as proxy for fp16 (fp16 GGUF >28GB caused filesystem issues on Lustre)
- **Infrastructure**: HSE University HPC cluster (cHARISMa), V100 32GB nodes

## Limitations

- Q4_K_M and Q5_K_M for 14B unavailable (fp16 GGUF conversion fails on 28GB+ files on Lustre)
- fp16 baseline via HF transformers available only for 14B (smaller HF models deleted before baseline eval)
- Single evaluation run per model (no variance estimate across different text samples)

## Citation

```bibtex
@misc{dashilovskiy2025quantboundary,
  title   = {Quantization Boundary Study: When Does PTQ Hurt Small LLMs?},
  author  = {Dashilovskiy},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/llm-quantization-boundary}
}
```

## Acknowledgements

This research was supported in part through computational resources of HPC facilities at HSE University.
