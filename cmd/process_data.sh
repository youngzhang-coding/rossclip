#!/bin/bash
# filepath: ~/rossclip/cmd/process_data.sh

# Exit on any error
set -e
PROJECT_ROOT="$HOME/rossclip"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export HF_ENDPOINT="https://hf-mirror.com"

# Set CUDA optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python scripts/process_data.py \
    --input_path $PROJECT_ROOT/dataset/data \
    --output_path $PROJECT_ROOT/rossclip/dataset/output/meta.json \
    --detector_id IDEA-Research/grounding-dino-tiny \
    --spacy_language_model_id en_core_web_sm \
    --device cuda:0 \
    --log_level INFO