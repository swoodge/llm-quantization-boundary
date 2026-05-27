#!/bin/bash
#SBATCH --job-name=eval_ppl
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/eval_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/eval_%j.err

module load gnu12/12.1
module load cmake/3.31.8
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

> ./results/ppl_all.txt

for SIZE in 0.5b 1.5b 3b 7b 14b; do
    for QTYPE in fp16 Q2_K Q3_K_M Q4_K_M Q5_K_M Q8_0; do
        MODEL="./models/qwen2.5-${SIZE}-${QTYPE}.gguf"

        if [ ! -f "$MODEL" ]; then
            echo "[skip] $MODEL not found" | tee -a ./results/ppl_all.txt
            continue
        fi

        echo "=== PPL: qwen2.5-${SIZE}-${QTYPE} ===" | tee -a ./results/ppl_all.txt
        ./llama.cpp/build_cpu/bin/llama-perplexity \
            -m "$MODEL" \
            -f ./data/wikitext2_test.txt \
            --ctx-size 512 \
            -t 16 \
            2>&1 | tee -a ./results/ppl_all.txt | grep "Final"
    done
done

echo "=== ALL DONE ===" | tee -a ./results/ppl_all.txt
