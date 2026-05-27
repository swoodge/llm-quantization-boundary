#!/bin/bash
#SBATCH --job-name=fix_14b_q4q5
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/fix_14b_from_q8_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/fix_14b_from_q8_%j.err

module load gnu12/12.1
module load cmake/3.31.8
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

# Q8_0 уже есть (15G) — реквантизуем из него в Q4 и Q5
echo "=== Q4_K_M from Q8_0 ==="
./llama.cpp/build_cpu/bin/llama-quantize \
    ./models/qwen2.5-14b-Q8_0.gguf \
    ./models/qwen2.5-14b-Q4_K_M.gguf \
    Q4_K_M
ls -lh ./models/qwen2.5-14b-Q4_K_M.gguf

echo "=== Q5_K_M from Q8_0 ==="
./llama.cpp/build_cpu/bin/llama-quantize \
    ./models/qwen2.5-14b-Q8_0.gguf \
    ./models/qwen2.5-14b-Q5_K_M.gguf \
    Q5_K_M
ls -lh ./models/qwen2.5-14b-Q5_K_M.gguf

echo "=== Final ==="
ls -lh ./models/qwen2.5-14b-*.gguf
