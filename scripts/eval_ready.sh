#!/bin/bash
#SBATCH --job-name=eval_ready
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/eval_ready_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/eval_ready_%j.err

module load gnu12/12.1
module load cmake/3.31.8
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment
> ./results/ppl_all.txt

for MODEL in ./models/*.gguf; do
    NAME=$(basename $MODEL .gguf)
    echo "=== PPL: $NAME ===" | tee -a ./results/ppl_all.txt
    ./llama.cpp/build_cpu/bin/llama-perplexity \
        -m "$MODEL" \
        -f ./data/wikitext2_test.txt \
        --ctx-size 512 \
        -t 16 \
        2>&1 | tee -a ./results/ppl_all.txt | grep "Final"
done

echo "=== ALL DONE ===" | tee -a ./results/ppl_all.txt
