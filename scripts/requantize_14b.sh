#!/bin/bash
#SBATCH --job-name=requantize_14b
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/requantize_14b_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/requantize_14b_%j.err

module load gnu12/12.1
module load cmake/3.31.8
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

echo "Disk space before:"
df -h ~ | tail -1

echo "=== Converting qwen2.5-14b ==="
python3 llama.cpp/convert_hf_to_gguf.py \
    ./hf_models/Qwen2.5-14B \
    --outfile ./models/qwen2.5-14b-fp16.gguf \
    --outtype f16

echo "=== Quantizing qwen2.5-14b ==="
for QTYPE in Q2_K Q3_K_M Q4_K_M Q5_K_M Q8_0; do
    ./llama.cpp/build_cpu/bin/llama-quantize \
        ./models/qwen2.5-14b-fp16.gguf \
        ./models/qwen2.5-14b-${QTYPE}.gguf \
        $QTYPE
    echo "Done: qwen2.5-14b-${QTYPE}.gguf"
done

rm ./models/qwen2.5-14b-fp16.gguf
echo "=== Done ==="
ls -lh ./models/qwen2.5-14b-*.gguf
