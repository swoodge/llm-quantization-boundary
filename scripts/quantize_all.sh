#!/bin/bash
#SBATCH --job-name=quantize_all
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/quantize_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/quantize_%j.err

module load gnu12/12.1
module load cmake/3.31.8
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

for SIZE in 1.5B 3B 7B 14B; do
    NAME="qwen2.5-${SIZE,,}"
    HF_PATH="./hf_models/Qwen2.5-${SIZE}"

    if [ ! -d "$HF_PATH" ]; then
        echo "[skip] $HF_PATH not found"
        continue
    fi

    echo "=== Converting $NAME ==="
    python3 llama.cpp/convert_hf_to_gguf.py \
        "$HF_PATH" \
        --outfile "./models/${NAME}-fp16.gguf" \
        --outtype f16

    echo "=== Quantizing $NAME ==="
    for QTYPE in Q2_K Q3_K_M Q4_K_M Q5_K_M Q8_0; do
        ./llama.cpp/build_cpu/bin/llama-quantize \
            "./models/${NAME}-fp16.gguf" \
            "./models/${NAME}-${QTYPE}.gguf" \
            $QTYPE
        echo "  Done: ${NAME}-${QTYPE}.gguf"
    done
done

echo "=== Quantization complete ==="
ls -lh ./models/*.gguf
