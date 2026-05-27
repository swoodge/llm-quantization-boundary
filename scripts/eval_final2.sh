#!/bin/bash
#SBATCH --job-name=eval_final2
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/eval_final2_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/eval_final2_%j.err

module load gnu12/12.1
module load cmake/3.31.8
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

# 1. Оценить 14B Q3_K_M через llama-perplexity (CPU — не нужен GPU)
echo "=== PPL: qwen2.5-14b-Q3_K_M ===" | tee -a ./results/ppl_all.txt
./llama.cpp/build_cpu/bin/llama-perplexity \
    -m ./models/qwen2.5-14b-Q3_K_M.gguf \
    -f ./data/wikitext2_test.txt \
    --ctx-size 512 -t 16 \
    2>&1 | tee -a ./results/ppl_all.txt | grep "Final"

echo "=== PPL: qwen2.5-14b-Q2_K ===" | tee -a ./results/ppl_all.txt
./llama.cpp/build_cpu/bin/llama-perplexity \
    -m ./models/qwen2.5-14b-Q2_K.gguf \
    -f ./data/wikitext2_test.txt \
    --ctx-size 512 -t 16 \
    2>&1 | tee -a ./results/ppl_all.txt | grep "Final"

echo "=== DONE ===" | tee -a ./results/ppl_all.txt
