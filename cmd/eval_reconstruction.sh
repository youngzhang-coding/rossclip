#!/bin/bash
# filepath: $HOME/rossclip/cmd/eval_reconstruction.sh

# Exit on any error
set -e
PROJECT_ROOT="$HOME/rossclip"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_$HOME:$PYTHONPATH"
export HF_ENDPOINT="https://hf-mirror.com"

python scripts/eval_reconstruction.py \
    --input_path $PROJECT_ROOT/dataset/data/000000003.jpg \
    --output_path $PROJECT_ROOT/dataset/output/reconstructed_000000003.jpg \
    --cpkt_path $PROJECT_ROOT/ross_clip/80kucboaw18h7nhd4fhax/checkpoints/epoch=49-step=3950.ckpt \
    --config_path $PROJECT_ROOT/src/config/config.yaml \
    --detector_id IDEA-Research/grounding-dino-base \
    --spacy_language_model_id en_core_web_sm    \
    --device cuda:0 \
    --weights_only false
