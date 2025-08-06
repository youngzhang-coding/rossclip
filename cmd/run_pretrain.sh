#!/bin/bash
# filepath: ~/rossclip/cmd/run_pretrain.sh
# Exit on any error
set -e
PROJECT_ROOT="$HOME/rossclip"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export HF_ENDPOINT="https://hf-mirror.com"

python scripts/train.py