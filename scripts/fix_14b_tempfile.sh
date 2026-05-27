#!/bin/bash
#SBATCH --job-name=fix_14b_tmp
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/fix_14b_tmp_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/fix_14b_tmp_%j.err

module load gnu12/12.1
module load cmake/3.31.8
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

rm -f ./models/qwen2.5-14b-fp16.gguf
rm -f ./models/qwen2.5-14b-Q4_K_M.gguf
rm -f ./models/qwen2.5-14b-Q5_K_M.gguf

echo "=== Converting 14b fp16 with --use-temp-file ==="
python3 llama.cpp/convert_hf_to_gguf.py \
    ./hf_models/Qwen2.5-14B \
    --outfile ./models/qwen2.5-14b-fp16.gguf \
    --outtype f16 \
    --use-temp-file

echo "Convert exit code: $?"
ls -lh ./models/qwen2.5-14b-fp16.gguf 2>/dev/null || echo "CONVERSION FAILED"

# Квантизовать только если конвертация прошла успешно
if [ -f "./models/qwen2.5-14b-fp16.gguf" ]; then
    SIZE=$(stat -c%s ./models/qwen2.5-14b-fp16.gguf)
    echo "fp16 size: $SIZE bytes"
    
    if [ $SIZE -gt 25000000000 ]; then
        echo "=== Quantizing Q4_K_M ==="
        ./llama.cpp/build_cpu/bin/llama-quantize \
            ./models/qwen2.5-14b-fp16.gguf \
            ./models/qwen2.5-14b-Q4_K_M.gguf \
            Q4_K_M

        echo "=== Quantizing Q5_K_M ==="
        ./llama.cpp/build_cpu/bin/llama-quantize \
            ./models/qwen2.5-14b-fp16.gguf \
            ./models/qwen2.5-14b-Q5_K_M.gguf \
            Q5_K_M

        rm ./models/qwen2.5-14b-fp16.gguf
    else
        echo "ERROR: fp16 file too small, conversion failed"
    fi
fi

echo "=== Final ==="
ls -lh ./models/qwen2.5-14b-*.gguf
