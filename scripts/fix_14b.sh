#!/bin/bash
#SBATCH --job-name=fix_14b
#SBATCH --account=proj_1782
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/dashilovskiy/quant_experiment/logs/fix_14b_%j.out
#SBATCH --error=/home/dashilovskiy/quant_experiment/logs/fix_14b_%j.err

module load gnu12/12.1
module load cmake/3.31.8
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

cd ~/quant_experiment

# Удалить повреждённые файлы
rm -f ./models/qwen2.5-14b-Q4_K_M.gguf
rm -f ./models/qwen2.5-14b-Q5_K_M.gguf
rm -f ./models/qwen2.5-14b-Q8_0.gguf

# Переконвертировать fp16 если нет
if [ ! -f "./models/qwen2.5-14b-fp16.gguf" ]; then
    echo "=== Converting qwen2.5-14b to fp16 ==="
    python3 llama.cpp/convert_hf_to_gguf.py \
        ./hf_models/Qwen2.5-14B \
        --outfile ./models/qwen2.5-14b-fp16.gguf \
        --outtype f16
fi

echo "fp16 size: $(ls -lh ./models/qwen2.5-14b-fp16.gguf | awk '{print $5}')"

# Квантизовать только три формата
for QTYPE in Q4_K_M Q5_K_M Q8_0; do
    echo "=== Quantizing $QTYPE ==="
    ./llama.cpp/build_cpu/bin/llama-quantize \
        ./models/qwen2.5-14b-fp16.gguf \
        ./models/qwen2.5-14b-${QTYPE}.gguf \
        $QTYPE
    echo "Size: $(ls -lh ./models/qwen2.5-14b-${QTYPE}.gguf | awk '{print $5}')"
done

rm -f ./models/qwen2.5-14b-fp16.gguf
echo "=== Done ==="
ls -lh ./models/qwen2.5-14b-*.gguf
