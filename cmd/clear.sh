#!/bin/bash

set -e
PROJECT_ROOT="$HOME/rossclip"
cd "$PROJECT_ROOT"

# Clear the contents of the checkpoints directory
rm -rf ./ross_clip

# Clear the contents of the training log directory
rm -rf ./train_log

# Clear the hydra output directory
rm -rf ./outputs

# Clear the contents of the detected results directory
rm -rf ./detected_results