#!/bin/bash
#SBATCH --job-name=fix_14b_q4q5
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/fix_14b_q4q5_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/fix_14b_q4q5_%j.err

module load gnu12/12.1
module load cmake/3.31.8
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

echo "=== Q4_K_M ==="
python3 llama.cpp/convert_hf_to_gguf.py \
    ./hf_models/Qwen2.5-14B \
    --outfile ./models/qwen2.5-14b-Q4_K_M.gguf \
    --outtype q4_k_m
echo "Exit code: $?"
ls -lh ./models/qwen2.5-14b-Q4_K_M.gguf 2>/dev/null || echo "FILE NOT CREATED"

echo "=== Q5_K_M ==="
python3 llama.cpp/convert_hf_to_gguf.py \
    ./hf_models/Qwen2.5-14B \
    --outfile ./models/qwen2.5-14b-Q5_K_M.gguf \
    --outtype q5_k_m
echo "Exit code: $?"
ls -lh ./models/qwen2.5-14b-Q5_K_M.gguf 2>/dev/null || echo "FILE NOT CREATED"

echo "=== Final ==="
ls -lh ./models/qwen2.5-14b-*.gguf
